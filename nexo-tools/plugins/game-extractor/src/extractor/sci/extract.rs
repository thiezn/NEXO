use anyhow::{Result, Context};
use crate::extractor::common::progress::ProgressBar;

use crate::extractor::common::metadata::LoraEntry;
use crate::extractor::common::output::{OutputManager, PaletteImage, save_image, write_jsonl, sanitize_name};
use crate::extractor::engine::ExtractionSummary;
use super::{palette, picture, resource, resource_map::ResourceType, sound, version, view};

/// Minimum number of non-black palette entries to consider a palette usable for views.
const MIN_PALETTE_COLORS: usize = 100;

/// Helper struct implementing PaletteImage for SCI views/pictures.
struct SciImage {
    width: u16,
    height: u16,
    pixels: Vec<u8>,
    palette: palette::Palette,
    transparent: u8,
}

impl PaletteImage for SciImage {
    fn width(&self) -> u16 { self.width }
    fn height(&self) -> u16 { self.height }
    fn pixels(&self) -> &[u8] { &self.pixels }
    fn palette_color(&self, index: u8) -> (u8, u8, u8) {
        if index == self.transparent {
            return (0, 0, 0); // transparent = black
        }
        let c = self.palette[index as usize];
        (c[0], c[1], c[2])
    }
}

pub fn extract_game(game: &version::SciGameInfo, output_root: &std::path::Path, progress: Option<&ProgressBar>) -> Result<ExtractionSummary> {
    let prefix = sanitize_name(&game.display_name);
    let mut log = Vec::new();

    log.push(format!("{}: loading resources ({})...", prefix, game.version));

    let res_mgr = resource::ResourceManager::new(game)
        .context("Failed to initialize resource manager")?;

    let counts = res_mgr.resource_counts();
    let mut sorted_counts: Vec<_> = counts.iter().collect();
    sorted_counts.sort_by_key(|(name, _)| *name);
    for (name, count) in &sorted_counts {
        log.push(format!("{}: {}: {}", prefix, name, count));
    }

    let output = OutputManager::new_for_game(output_root, &game.display_name)?;
    output.ensure_dirs()?;
    output.ensure_lora_dirs()?;
    log.push(format!("{}: output directory: {}", prefix, output.base_dir().display()));

    // Load global palette
    let mut global_palette = load_global_palette(&res_mgr, game.version);

    // Extract pictures (backgrounds) and collect the best palette for views
    let pic_numbers = res_mgr.list_resources(ResourceType::Pic);
    if let Some(pb) = progress {
        pb.set_message(format!("{} pics", pic_numbers.len()));
    }
    let mut best_pic_palette: Option<palette::Palette> = None;
    let pic_results: Vec<_> = pic_numbers.iter()
        .filter_map(|&num| {
            let result = match extract_pic(&res_mgr, &output, num, game.version, &global_palette) {
                Ok((count, lora, pic_pal)) => {
                    if best_pic_palette.is_none() {
                        if let Some(pal) = pic_pal {
                            let non_black = pal.iter().filter(|c| c[0] != 0 || c[1] != 0 || c[2] != 0).count();
                            if non_black > MIN_PALETTE_COLORS {
                                best_pic_palette = Some(pal);
                            }
                        }
                    }
                    Some((count, lora))
                }
                Err(e) => {
                    log.push(format!("{}: pic_{:03}: FAILED: {}", prefix, num, e));
                    None
                }
            };
            if let Some(pb) = progress { pb.tick(); }
            result
        })
        .collect();

    // If global palette is sparse, upgrade it with the best pic palette
    let global_non_black = global_palette.iter().filter(|c| c[0] != 0 || c[1] != 0 || c[2] != 0).count();
    if global_non_black < MIN_PALETTE_COLORS {
        if let Some(pic_pal) = best_pic_palette {
            log.push(format!("{}: using pic palette for views ({} -> {} non-black colors)",
                prefix, global_non_black,
                pic_pal.iter().filter(|c| c[0] != 0 || c[1] != 0 || c[2] != 0).count()));
            global_palette = pic_pal;
        }
    }

    let mut lora_backgrounds: Vec<LoraEntry> = Vec::new();
    let mut total_pics = 0;
    for (count, lora) in pic_results {
        total_pics += count;
        lora_backgrounds.extend(lora);
    }

    // Extract views (sprites)
    let view_numbers = res_mgr.list_resources(ResourceType::View);
    if let Some(pb) = progress {
        pb.set_message(format!("{} views", view_numbers.len()));
    }
    let view_results: Vec<_> = view_numbers.iter()
        .filter_map(|&num| {
            let result = match extract_view(&res_mgr, &output, num, game.version, &global_palette) {
                Ok(r) => Some(r),
                Err(e) => {
                    log.push(format!("{}: view_{:03}: FAILED: {}", prefix, num, e));
                    None
                }
            };
            if let Some(pb) = progress { pb.tick(); }
            result
        })
        .collect();

    let mut lora_sprites: Vec<LoraEntry> = Vec::new();
    let mut total_views = 0;
    for (count, lora) in view_results {
        total_views += count;
        lora_sprites.extend(lora);
    }

    // Extract sounds
    let sound_numbers = res_mgr.list_resources(ResourceType::Sound);
    if let Some(pb) = progress {
        pb.set_message(format!("{} sounds", sound_numbers.len()));
    }
    let sound_results: Vec<_> = sound_numbers.iter()
        .filter_map(|&num| {
            let result = extract_sound(&res_mgr, &output, num).ok();
            if let Some(pb) = progress { pb.tick(); }
            result
        })
        .collect();

    let total_sounds: usize = sound_results.iter().sum();

    log.push(format!("{}: extracted {} backgrounds, {} view sprites, {} sounds",
        prefix, total_pics, total_views, total_sounds));

    // Write LoRA metadata
    if !lora_backgrounds.is_empty() {
        write_jsonl(&lora_backgrounds, &output.lora_backgrounds_dir().join("metadata.jsonl"))?;
        log.push(format!("{}: lora training: {} background images", prefix, lora_backgrounds.len()));
    }
    if !lora_sprites.is_empty() {
        write_jsonl(&lora_sprites, &output.lora_sprites_dir().join("metadata.jsonl"))?;
        log.push(format!("{}: lora training: {} sprite sheets", prefix, lora_sprites.len()));
    }

    Ok(ExtractionSummary {
        game_name: game.display_name.clone(),
        log_lines: log,
        backgrounds: total_pics,
        objects: 0,
        sounds: total_sounds,
        sprites: total_views,
        speech_files: 0,
    })
}

/// Load the global palette for the game.
fn load_global_palette(res_mgr: &resource::ResourceManager, version: version::SciVersion) -> palette::Palette {
    // Try palette 999 first (common default), then 0, then EGA fallback
    for pal_num in [999, 0, 1] {
        if let Ok(data) = res_mgr.get_resource(ResourceType::Palette, pal_num) {
            if let Ok(pal) = palette::parse_palette(&data, version) {
                return pal;
            }
        }
    }

    // Fallback to EGA palette
    palette::default_ega_palette()
}

/// Extract a single picture resource.
fn extract_pic(
    res_mgr: &resource::ResourceManager,
    output: &OutputManager,
    number: u16,
    version: version::SciVersion,
    global_palette: &palette::Palette,
) -> Result<(usize, Vec<LoraEntry>, Option<palette::Palette>)> {
    let data = res_mgr.get_resource(ResourceType::Pic, number)?;

    let pic = picture::parse_picture(&data, version, global_palette)?;

    let returned_palette = pic.palette.clone();
    let pal = pic.palette.as_ref().unwrap_or(global_palette);

    let pic_dir = output.base_dir().join("assets").join("pics").join(format!("pic_{:03}", number));
    std::fs::create_dir_all(&pic_dir)?;

    let img = SciImage {
        width: pic.width,
        height: pic.height,
        pixels: pic.pixels,
        palette: *pal,
        transparent: 255, // no transparency for backgrounds
    };

    let bg_path = pic_dir.join("background.png");
    save_image(&img, &bg_path)?;

    // LoRA training copy
    let lora_filename = format!("pic_{:03}.png", number);
    let lora_path = output.lora_backgrounds_images_dir().join(&lora_filename);
    save_image(&img, &lora_path)?;

    let lora_entry = LoraEntry {
        image: format!("images/{}", lora_filename),
        text: format!("a pixel art scene, background pic {}", number),
    };

    Ok((1, vec![lora_entry], returned_palette))
}

/// Extract a single view resource (all loops/cels).
fn extract_view(
    res_mgr: &resource::ResourceManager,
    output: &OutputManager,
    number: u16,
    version: version::SciVersion,
    global_palette: &palette::Palette,
) -> Result<(usize, Vec<LoraEntry>)> {
    let data = res_mgr.get_resource(ResourceType::View, number)?;

    let view_res = view::parse_view(&data, version)
        .with_context(|| format!("Failed to parse view {} ({} bytes)", number, data.len()))?;

    let pal = view_res.embedded_palette.as_ref().unwrap_or(global_palette);

    let view_dir = output.base_dir().join("assets").join("views").join(format!("view_{:03}", number));
    std::fs::create_dir_all(&view_dir)?;

    let mut total_cels = 0;
    let mut lora_entries = Vec::new();

    for (loop_idx, vloop) in view_res.loops.iter().enumerate() {
        if vloop.cels.is_empty() {
            continue;
        }

        let loop_dir = view_dir.join(format!("loop_{:02}", loop_idx));
        std::fs::create_dir_all(&loop_dir)?;

        // Save individual cels
        for (cel_idx, cel) in vloop.cels.iter().enumerate() {
            if cel.width == 0 || cel.height == 0 {
                continue;
            }

            let img = SciImage {
                width: cel.width,
                height: cel.height,
                pixels: cel.pixels.clone(),
                palette: *pal,
                transparent: cel.clear_key,
            };

            let cel_path = loop_dir.join(format!("cel_{:02}.png", cel_idx));
            let _ = save_image(&img, &cel_path);
            total_cels += 1;
        }

        // Build sprite sheet for this loop
        let cel_refs: Vec<&view::ViewCel> = vloop.cels.iter()
            .filter(|c| c.width > 0 && c.height > 0)
            .collect();

        if !cel_refs.is_empty() {
            let clear_key = cel_refs[0].clear_key;
            let (sw, sh, sheet_pixels) = view::build_sprite_sheet(&cel_refs, clear_key);

            if sw > 0 && sh > 0 {
                let sheet_img = SciImage {
                    width: sw as u16,
                    height: sh as u16,
                    pixels: sheet_pixels,
                    palette: *pal,
                    transparent: clear_key,
                };

                let sheet_path = view_dir.join(format!("loop_{:02}_sheet.png", loop_idx));
                let _ = save_image(&sheet_img, &sheet_path);

                // LoRA training copy
                let lora_filename = format!("view_{:03}_loop_{:02}.png", number, loop_idx);
                let lora_path = output.lora_sprites_images_dir().join(&lora_filename);
                let _ = save_image(&sheet_img, &lora_path);

                lora_entries.push(LoraEntry {
                    image: format!("images/{}", lora_filename),
                    text: format!("a pixel art sprite sheet, view {} loop {} animation", number, loop_idx),
                });
            }
        }
    }

    Ok((total_cels, lora_entries))
}

/// Extract a single sound resource.
fn extract_sound(
    res_mgr: &resource::ResourceManager,
    output: &OutputManager,
    number: u16,
) -> Result<usize> {
    let data = res_mgr.get_resource(ResourceType::Sound, number)?;

    let sounds = sound::extract_sound(&data, number)?;

    let audio_dir = output.base_dir().join("assets").join("audio");
    std::fs::create_dir_all(&audio_dir)?;

    let mut count = 0;
    for snd in &sounds {
        let filename = format!("sound_{:03}.{}", number, snd.format.extension());
        let path = audio_dir.join(&filename);
        std::fs::write(&path, &snd.data)?;
        count += 1;
    }

    Ok(count)
}
