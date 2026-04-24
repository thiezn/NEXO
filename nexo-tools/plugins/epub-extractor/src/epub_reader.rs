use crate::cli::ImageMode;
use crate::models::*;
use base64::Engine;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

pub struct EpubContent {
    pub metadata: BookMetadata,
    pub toc_labels: Vec<String>,
    pub chapters: Vec<RawChapter>,
    /// Image binary data keyed by unique filename (only populated in Files mode)
    pub image_file_data: HashMap<String, Vec<u8>>,
}

pub struct RawChapter {
    pub paragraphs: Vec<Paragraph>,
}

struct ManifestItem {
    href: String,
    media_type: String,
    properties: Option<String>,
}

pub fn read_epub(path: &Path, image_mode: ImageMode) -> cli_helpers::Result<EpubContent> {
    let file = std::fs::File::open(path)
        .map_err(|e| cli_helpers::Error::Io(format!("Failed to open '{}': {e}", path.display())))?;

    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| cli_helpers::Error::Other(format!("Invalid ZIP/EPUB: {e}")))?;

    let opf_path = parse_container_xml(&mut archive)?;
    let opf_dir = Path::new(&opf_path)
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();

    let opf_content = read_zip_text(&mut archive, &opf_path)?;
    let opf = parse_opf(&opf_content, &opf_dir)?;

    let toc_labels = parse_toc(&mut archive, &opf)?;

    let mut image_file_data: HashMap<String, Vec<u8>> = HashMap::new();
    let mut base64_cache: HashMap<String, String> = HashMap::new();

    let image_manifest: HashMap<String, (&str, &str)> = opf
        .manifest
        .iter()
        .filter(|(_, item)| item.media_type.starts_with("image/"))
        .map(|(id, item)| (item.href.clone(), (id.as_str(), item.media_type.as_str())))
        .collect();

    let mut chapters = Vec::with_capacity(opf.spine.len());

    for itemref in &opf.spine {
        let Some(item) = opf.manifest.get(itemref) else {
            tracing::warn!("Spine references unknown manifest item: {itemref}");
            chapters.push(RawChapter {
                paragraphs: Vec::new(),
            });
            continue;
        };

        let xhtml_path = item.href.clone();

        let xhtml_content = match read_zip_text(&mut archive, &xhtml_path) {
            Ok(content) => content,
            Err(e) => {
                tracing::warn!("Failed to read spine item '{xhtml_path}': {e}");
                chapters.push(RawChapter {
                    paragraphs: Vec::new(),
                });
                continue;
            }
        };

        let xhtml_dir = Path::new(&xhtml_path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        let mut ctx = ChapterParseCtx {
            xhtml_dir: &xhtml_dir,
            image_manifest: &image_manifest,
            image_mode,
            archive: &mut archive,
            image_file_data: &mut image_file_data,
            base64_cache: &mut base64_cache,
        };

        let paragraphs = parse_xhtml_chapter(&xhtml_content, &mut ctx)?;
        chapters.push(RawChapter { paragraphs });
    }

    Ok(EpubContent {
        metadata: opf.metadata,
        toc_labels,
        chapters,
        image_file_data,
    })
}

// ── Container XML ──────────────────────────────────────────────────────

fn parse_container_xml(
    archive: &mut zip::ZipArchive<std::fs::File>,
) -> cli_helpers::Result<String> {
    let xml = read_zip_text(archive, "META-INF/container.xml")?;
    let doc = parse_xml(&xml)
        .map_err(|e| cli_helpers::Error::Other(format!("Invalid container.xml: {e}")))?;

    doc.descendants()
        .find(|n| n.tag_name().name() == "rootfile")
        .and_then(|n| n.attribute("full-path"))
        .map(String::from)
        .ok_or_else(|| cli_helpers::Error::Other("No rootfile found in container.xml".to_string()))
}

// ── OPF Parsing ────────────────────────────────────────────────────────

struct OpfData {
    metadata: BookMetadata,
    manifest: HashMap<String, ManifestItem>,
    spine: Vec<String>,
    toc_id: Option<String>,
}

fn parse_opf(content: &str, opf_dir: &str) -> cli_helpers::Result<OpfData> {
    let doc =
        parse_xml(content).map_err(|e| cli_helpers::Error::Other(format!("Invalid OPF: {e}")))?;

    let metadata = extract_metadata(&doc);
    let manifest = extract_manifest(&doc, opf_dir);

    let spine_node = doc.descendants().find(|n| n.tag_name().name() == "spine");

    let toc_id = spine_node.and_then(|n| n.attribute("toc").map(String::from));

    let spine: Vec<String> = spine_node
        .map(|node| {
            node.children()
                .filter(|n| n.tag_name().name() == "itemref")
                .filter_map(|n| n.attribute("idref").map(String::from))
                .collect()
        })
        .unwrap_or_default();

    Ok(OpfData {
        metadata,
        manifest,
        spine,
        toc_id,
    })
}

fn extract_metadata(doc: &roxmltree::Document) -> BookMetadata {
    let find = |local_name: &str| -> Option<String> {
        doc.descendants()
            .find(|n| n.tag_name().name() == local_name && n.is_element())
            .and_then(|n| n.text())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };

    BookMetadata {
        title: find("title"),
        author: find("creator"),
        publisher: find("publisher"),
        language: find("language"),
        identifier: find("identifier"),
        date: find("date"),
        rights: find("rights"),
    }
}

fn extract_manifest(doc: &roxmltree::Document, opf_dir: &str) -> HashMap<String, ManifestItem> {
    doc.descendants()
        .filter(|n| n.tag_name().name() == "item")
        .filter_map(|n| {
            let id = n.attribute("id")?.to_string();
            let raw_href = n.attribute("href")?;
            let media_type = n.attribute("media-type").unwrap_or("").to_string();
            let properties = n.attribute("properties").map(String::from);

            let href = resolve_relative(opf_dir, raw_href);
            let href = normalize_path(&href);

            Some((
                id,
                ManifestItem {
                    href,
                    media_type,
                    properties,
                },
            ))
        })
        .collect()
}

// ── TOC Parsing ────────────────────────────────────────────────────────

fn parse_toc(
    archive: &mut zip::ZipArchive<std::fs::File>,
    opf: &OpfData,
) -> cli_helpers::Result<Vec<String>> {
    // Try NCX first (EPUB2)
    if let Some(toc_id) = &opf.toc_id {
        if let Some(item) = opf.manifest.get(toc_id) {
            if let Ok(ncx_content) = read_zip_text(archive, &item.href) {
                return parse_ncx(&ncx_content);
            }
        }
    }

    // Try nav document (EPUB3) — check manifest properties first
    if let Some((_, item)) = opf.manifest.iter().find(|(_, item)| {
        item.properties
            .as_deref()
            .is_some_and(|p| p.split_whitespace().any(|v| v == "nav"))
    }) {
        if let Ok(content) = read_zip_text(archive, &item.href) {
            return parse_nav_toc(&content);
        }
    }

    Ok(Vec::new())
}

fn parse_ncx(content: &str) -> cli_helpers::Result<Vec<String>> {
    let content = clean_entities(content);
    let doc =
        parse_xml(&content).map_err(|e| cli_helpers::Error::Other(format!("Invalid NCX: {e}")))?;

    let mut labels = Vec::new();
    collect_ncx_labels(&doc.root(), &mut labels);
    Ok(labels)
}

fn collect_ncx_labels(node: &roxmltree::Node, out: &mut Vec<String>) {
    if node.tag_name().name() == "navPoint" {
        // Only search immediate children for navLabel > text, not descendants,
        // to avoid picking up text from nested navPoints.
        for child in node.children() {
            if child.tag_name().name() == "navLabel" {
                if let Some(text_node) = child.children().find(|n| n.tag_name().name() == "text") {
                    if let Some(text) = text_node.text() {
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            out.push(trimmed.to_string());
                        }
                    }
                }
                break;
            }
        }
        for child in node.children() {
            if child.tag_name().name() == "navPoint" {
                collect_ncx_labels(&child, out);
            }
        }
    } else {
        for child in node.children() {
            collect_ncx_labels(&child, out);
        }
    }
}

fn parse_nav_toc(content: &str) -> cli_helpers::Result<Vec<String>> {
    let content = clean_entities(content);
    let doc = parse_xml(&content)
        .map_err(|e| cli_helpers::Error::Other(format!("Invalid nav XHTML: {e}")))?;

    let mut labels = Vec::new();

    let nav = doc.descendants().find(|n| {
        n.tag_name().name() == "nav"
            && n.attributes()
                .any(|a| a.name() == "type" && a.value() == "toc")
    });

    if let Some(nav_node) = nav {
        collect_nav_links(&nav_node, &mut labels);
    }

    Ok(labels)
}

fn collect_nav_links(node: &roxmltree::Node, out: &mut Vec<String>) {
    for child in node.descendants() {
        if child.tag_name().name() == "a" {
            let text: String = child
                .descendants()
                .filter(|n| n.is_text())
                .filter_map(|n| n.text())
                .collect::<Vec<_>>()
                .join(" ");
            let trimmed = text.trim().to_string();
            if !trimmed.is_empty() {
                out.push(trimmed);
            }
        }
    }
}

// ── XHTML Chapter Parsing ──────────────────────────────────────────────

const BLOCK_ELEMENTS: &[&str] = &[
    "p",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "li",
    "section",
    "article",
    "figure",
    "figcaption",
    "header",
    "footer",
    "pre",
    "table",
    "tr",
    "td",
    "th",
    "dl",
    "dt",
    "dd",
    "aside",
    "nav",
    "hr",
];

struct ChapterParseCtx<'a> {
    xhtml_dir: &'a str,
    image_manifest: &'a HashMap<String, (&'a str, &'a str)>,
    image_mode: ImageMode,
    archive: &'a mut zip::ZipArchive<std::fs::File>,
    image_file_data: &'a mut HashMap<String, Vec<u8>>,
    base64_cache: &'a mut HashMap<String, String>,
}

fn parse_xhtml_chapter(
    content: &str,
    ctx: &mut ChapterParseCtx,
) -> cli_helpers::Result<Vec<Paragraph>> {
    let content = clean_entities(content);
    let doc = parse_xml(&content)
        .map_err(|e| cli_helpers::Error::Other(format!("Invalid XHTML: {e}")))?;

    let body = doc
        .descendants()
        .find(|n| n.tag_name().name() == "body")
        .unwrap_or_else(|| doc.root());

    let mut paragraphs = Vec::new();
    let mut current_text = Vec::new();
    let mut current_images: Vec<ImageRef> = Vec::new();

    walk_node(
        &body,
        ctx,
        &mut paragraphs,
        &mut current_text,
        &mut current_images,
    );

    flush_paragraph(&mut paragraphs, &mut current_text, &mut current_images);

    Ok(paragraphs)
}

fn walk_node(
    node: &roxmltree::Node,
    ctx: &mut ChapterParseCtx,
    paragraphs: &mut Vec<Paragraph>,
    current_text: &mut Vec<String>,
    current_images: &mut Vec<ImageRef>,
) {
    for child in node.children() {
        if child.is_text() {
            if let Some(text) = child.text() {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    current_text.push(trimmed.to_string());
                }
            }
            continue;
        }

        if !child.is_element() {
            continue;
        }

        let tag = child.tag_name().name();

        if tag == "img" || tag == "image" {
            if ctx.image_mode == ImageMode::None {
                continue;
            }

            let src = child
                .attribute("src")
                .or_else(|| child.attribute("href"))
                .or_else(|| {
                    child
                        .attributes()
                        .find(|a| a.name() == "href")
                        .map(|a| a.value())
                });

            if let Some(src) = src {
                if let Some(image_ref) = resolve_image(src, ctx) {
                    current_images.push(image_ref);
                }
            }
            continue;
        }

        let is_block = BLOCK_ELEMENTS.contains(&tag);

        if is_block {
            flush_paragraph(paragraphs, current_text, current_images);
        }

        walk_node(&child, ctx, paragraphs, current_text, current_images);

        if is_block {
            flush_paragraph(paragraphs, current_text, current_images);
        }
    }
}

fn flush_paragraph(
    paragraphs: &mut Vec<Paragraph>,
    text: &mut Vec<String>,
    images: &mut Vec<ImageRef>,
) {
    if text.is_empty() && images.is_empty() {
        return;
    }

    paragraphs.push(Paragraph {
        text: std::mem::take(text),
        images: std::mem::take(images),
    });
}

fn resolve_image(src: &str, ctx: &mut ChapterParseCtx) -> Option<ImageRef> {
    let resolved = resolve_relative(ctx.xhtml_dir, src);
    let resolved = normalize_path(&resolved);

    let (id, media_type) = ctx.image_manifest.get(&resolved)?;

    let filename = Path::new(&resolved)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| resolved.clone());

    let path = format!("images/{filename}");

    let data = match ctx.image_mode {
        ImageMode::Base64 => {
            if let Some(cached) = ctx.base64_cache.get(&resolved) {
                Some(cached.clone())
            } else {
                let bytes = read_zip_bytes(ctx.archive, &resolved).ok()?;
                let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
                ctx.base64_cache.insert(resolved.clone(), encoded.clone());
                Some(encoded)
            }
        }
        ImageMode::Files => {
            if !ctx.image_file_data.contains_key(&filename) {
                if let Ok(bytes) = read_zip_bytes(ctx.archive, &resolved) {
                    ctx.image_file_data.insert(filename.clone(), bytes);
                }
            }
            None
        }
        ImageMode::None => None,
    };

    Some(ImageRef {
        path,
        id: id.to_string(),
        media_type: media_type.to_string(),
        data,
    })
}

// ── Utility functions ──────────────────────────────────────────────────

fn read_zip_text(
    archive: &mut zip::ZipArchive<std::fs::File>,
    path: &str,
) -> cli_helpers::Result<String> {
    let mut file = archive
        .by_name(path)
        .map_err(|e| cli_helpers::Error::Io(format!("ZIP entry '{path}' not found: {e}")))?;

    let mut content = String::new();
    file.read_to_string(&mut content)
        .map_err(|e| cli_helpers::Error::Io(format!("Failed to read '{path}': {e}")))?;

    Ok(content)
}

fn read_zip_bytes(
    archive: &mut zip::ZipArchive<std::fs::File>,
    path: &str,
) -> cli_helpers::Result<Vec<u8>> {
    let mut file = archive
        .by_name(path)
        .map_err(|e| cli_helpers::Error::Io(format!("ZIP entry '{path}' not found: {e}")))?;

    let mut buf = Vec::with_capacity(file.size() as usize);
    file.read_to_end(&mut buf)
        .map_err(|e| cli_helpers::Error::Io(format!("Failed to read '{path}': {e}")))?;

    Ok(buf)
}

/// Replace common HTML entities that roxmltree doesn't recognize.
fn clean_entities(content: &str) -> String {
    content
        .replace("&nbsp;", "\u{00A0}")
        .replace("&mdash;", "\u{2014}")
        .replace("&ndash;", "\u{2013}")
        .replace("&lsquo;", "\u{2018}")
        .replace("&rsquo;", "\u{2019}")
        .replace("&ldquo;", "\u{201C}")
        .replace("&rdquo;", "\u{201D}")
        .replace("&hellip;", "\u{2026}")
        .replace("&copy;", "\u{00A9}")
        .replace("&reg;", "\u{00AE}")
        .replace("&trade;", "\u{2122}")
        .replace("&euro;", "\u{20AC}")
        .replace("&pound;", "\u{00A3}")
        .replace("&yen;", "\u{00A5}")
        .replace("&cent;", "\u{00A2}")
}

fn parse_xml(content: &str) -> Result<roxmltree::Document<'_>, roxmltree::Error> {
    roxmltree::Document::parse_with_options(
        content,
        roxmltree::ParsingOptions {
            allow_dtd: true,
            ..Default::default()
        },
    )
}

fn resolve_relative(dir: &str, path: &str) -> String {
    if dir.is_empty() {
        path.to_string()
    } else {
        format!("{dir}/{path}")
    }
}

/// Normalize a path by resolving `.` and `..` segments.
fn normalize_path(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    parts.join("/")
}
