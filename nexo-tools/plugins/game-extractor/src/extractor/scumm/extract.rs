use std::collections::HashSet;
use anyhow::{Result, Context};
use rayon::prelude::*;

use crate::extractor::common::decrypt;
use crate::extractor::common::metadata::*;
use crate::extractor::common::output::{OutputManager, save_image, write_jsonl, sanitize_name};
use crate::extractor::common::progress::ProgressBar;
use crate::extractor::engine::ExtractionSummary;
use super::{block, costume, index, monster_sou, resource, room, sound, version};
use version::ScummVersion;

/// Per-room extraction result, collected after parallel processing.
struct RoomResult {
    bg_count: usize,
    obj_count: usize,
    sound_count: usize,
    sprite_count: usize,
    lora_backgrounds: Vec<LoraEntry>,
    lora_objects: Vec<LoraEntry>,
    lora_sprites: Vec<LoraEntry>,
    log_lines: Vec<String>,
}

pub fn extract_game(game: &version::GameInfo, output_root: &std::path::Path, progress: Option<&ProgressBar>) -> Result<ExtractionSummary> {
    let prefix = sanitize_name(&game.display_name);
    match game.version {
        ScummVersion::V3 | ScummVersion::V4 => {
            let output = OutputManager::new(output_root, &game.display_name, Default::default())?;
            output.ensure_dirs()?;
            let mut summary = ExtractionSummary {
                game_name: game.display_name.clone(),
                log_lines: vec![format!("{}: output directory: {}", prefix, output.base_dir().display())],
                backgrounds: 0, objects: 0, sounds: 0, sprites: 0, speech_files: 0,
            };
            extract_v3(game, &output, &prefix, &mut summary, progress)?;
            Ok(summary)
        }
        ScummVersion::V5 | ScummVersion::V6 | ScummVersion::V7 => {
            extract_v5v6v7(game, output_root, &prefix, progress)
        }
    }
}

/// Extract V5/V6/V7 games (block-based .000/.001 or .LA0/.LA1)
fn extract_v5v6v7(game: &version::GameInfo, output_root: &std::path::Path, prefix: &str, progress: Option<&ProgressBar>) -> Result<ExtractionSummary> {
    let mut log = Vec::new();

    // Read and decrypt index file
    let mut index_data = std::fs::read(&game.index_path)
        .context("Failed to read index file")?;
    if game.xor_key != 0 {
        decrypt::decrypt(&mut index_data, game.xor_key);
    }

    let game_index = index::parse_index(&index_data)
        .context("Failed to parse index")?;

    let room_name_count = game_index.room_names.len();
    log.push(format!("{}: found {} rooms, {} sounds in index{}",
        prefix,
        game_index.room_directory.len(),
        game_index.sound_directory.len(),
        if room_name_count > 0 { format!(", {} room names", room_name_count) } else { String::new() }));

    let output = OutputManager::new(output_root, &game.display_name, game_index.room_names)?;
    output.ensure_dirs()?;
    output.ensure_lora_dirs()?;
    log.push(format!("{}: output directory: {}", prefix, output.base_dir().display()));

    // Read and decrypt data file
    let mut data = std::fs::read(&game.data_path)
        .context("Failed to read data file")?;
    if game.xor_key != 0 {
        decrypt::decrypt(&mut data, game.xor_key);
    }

    let lflf_entries = resource::parse_data_file(&data)
        .context("Failed to parse data file")?;
    log.push(format!("{}: found {} LFLF entries", prefix, lflf_entries.len()));

    // Process rooms in parallel
    if let Some(pb) = progress {
        pb.set_message(format!("{} rooms", lflf_entries.len()));
    }
    let results: Vec<RoomResult> = lflf_entries.par_iter()
        .filter_map(|entry| {
            let result = extract_single_room(&data, entry, &output, game.version).ok();
            if let Some(pb) = progress { pb.tick(); }
            result
        })
        .collect();

    // Aggregate results
    let mut total_bg = 0;
    let mut total_obj = 0;
    let mut total_sounds = 0;
    let mut total_sprites = 0;
    let mut lora_backgrounds: Vec<LoraEntry> = Vec::new();
    let mut lora_objects: Vec<LoraEntry> = Vec::new();
    let mut lora_sprites: Vec<LoraEntry> = Vec::new();

    for result in results {
        for line in &result.log_lines {
            log.push(format!("{}: {}", prefix, line));
        }
        total_bg += result.bg_count;
        total_obj += result.obj_count;
        total_sounds += result.sound_count;
        total_sprites += result.sprite_count;
        lora_backgrounds.extend(result.lora_backgrounds);
        lora_objects.extend(result.lora_objects);
        lora_sprites.extend(result.lora_sprites);
    }

    log.push(format!("{}: extracted {} backgrounds, {} object images, {} sound files, {} sprites",
        prefix, total_bg, total_obj, total_sounds, total_sprites));

    // Write lora_training JSONL metadata files
    if !lora_backgrounds.is_empty() {
        write_jsonl(&lora_backgrounds, &output.lora_backgrounds_dir().join("metadata.jsonl"))?;
        log.push(format!("{}: lora training: {} background images", prefix, lora_backgrounds.len()));
    }
    if !lora_objects.is_empty() {
        write_jsonl(&lora_objects, &output.lora_objects_dir().join("metadata.jsonl"))?;
        log.push(format!("{}: lora training: {} object images", prefix, lora_objects.len()));
    }
    if !lora_sprites.is_empty() {
        write_jsonl(&lora_sprites, &output.lora_sprites_dir().join("metadata.jsonl"))?;
        log.push(format!("{}: lora training: {} sprite sheets", prefix, lora_sprites.len()));
    }

    // Extract speech from MONSTER.SOU if present
    let mut speech_files = 0;
    if let Some(ref sou_path) = game.sound_file {
        if let Some(pb) = progress {
            pb.set_message("speech...");
        }
        log.push(format!("{}: extracting speech from {}...", prefix, sou_path.file_name().unwrap().to_string_lossy()));
        speech_files = extract_speech(sou_path, &output, prefix, &mut log)?;
    }

    Ok(ExtractionSummary {
        game_name: game.display_name.clone(),
        log_lines: log,
        backgrounds: total_bg,
        objects: total_obj,
        sounds: total_sounds,
        sprites: total_sprites,
        speech_files,
    })
}

/// Extract a single room, returning all results for aggregation.
fn extract_single_room(
    data: &[u8],
    entry: &resource::LflfEntry,
    output: &OutputManager,
    version: ScummVersion,
) -> Result<RoomResult> {
    let room_num = entry.room_num as u16;

    let room_block = block::find_child(data, &entry.block, b"ROOM")
        .ok_or_else(|| anyhow::anyhow!("No ROOM block"))?;

    let header = room::get_room_header(data, &room_block, version)?;

    let palette_type = room::get_palette_type(data, &room_block);
    let room_folder = output.room_folder_name(room_num);

    // create_dir_all is idempotent, safe to call concurrently
    output.ensure_room_dirs(room_num)?;

    let mut result = RoomResult {
        bg_count: 0,
        obj_count: 0,
        sound_count: 0,
        sprite_count: 0,
        lora_backgrounds: Vec::new(),
        lora_objects: Vec::new(),
        lora_sprites: Vec::new(),
        log_lines: Vec::new(),
    };

    let mut obj_meta_list: Vec<ObjectMetadataJson> = Vec::new();
    let mut room_sound_refs: Vec<SoundRef> = Vec::new();

    // Extract background image
    match room::extract_background(data, &room_block, version) {
        Ok(Some(img)) => {
            if img.width > 0 && img.height > 0 {
                let path = output.room_image_dir(room_num).join("background.png");
                match save_image(&img, &path) {
                    Ok(_) => {
                        result.log_lines.push(format!("  {}: background {}x{}", room_folder, img.width, img.height));
                        result.bg_count += 1;

                        let lora_filename = format!("{}.png", room_folder);
                        let lora_path = output.lora_backgrounds_images_dir().join(&lora_filename);
                        let _ = save_image(&img, &lora_path);
                        result.lora_backgrounds.push(LoraEntry {
                            image: format!("images/{}", lora_filename),
                            text: format!("a pixel art scene of {}", room_folder.replace('_', " ")),
                        });
                    }
                    Err(e) => result.log_lines.push(format!("  Warning: {} background save failed: {}", room_folder, e)),
                }
            }
        }
        Ok(None) => {}
        Err(e) => result.log_lines.push(format!("  Warning: {} background failed: {}", room_folder, e)),
    }

    // Extract object metadata first
    let obj_metas = room::extract_object_metadata(data, &room_block, version);

    // Extract object images
    match room::extract_objects(data, &room_block, version) {
        Ok(objects) => {
            for obj in &objects {
                if obj.image.width > 0 && obj.image.height > 0 {
                    let obj_name = obj_metas.iter()
                        .find(|m| m.obj_id == obj.obj_id)
                        .and_then(|m| m.name.as_deref());

                    let obj_dir_name = match obj_name {
                        Some(name) if !name.is_empty() => name.to_string(),
                        _ => format!("obj_{:05}", obj.obj_id),
                    };

                    output.ensure_room_object_dir(room_num, &obj_dir_name)?;
                    let filename = format!("state_{:02}.png", obj.state);
                    let path = output.room_object_dir(room_num, &obj_dir_name).join(&filename);
                    if let Err(e) = save_image(&obj.image, &path) {
                        result.log_lines.push(format!("  Warning: {} obj {} save failed: {}", room_folder, obj.obj_id, e));
                    } else {
                        result.obj_count += 1;

                        let lora_obj_name = sanitize_name(&obj_dir_name);
                        let lora_filename = format!("{}_{}_state_{:02}.png",
                            room_folder, lora_obj_name, obj.state);
                        let lora_path = output.lora_objects_images_dir().join(&lora_filename);
                        let _ = save_image(&obj.image, &lora_path);
                        let description = match obj_name {
                            Some(name) if !name.is_empty() =>
                                format!("a pixel art object: {}", name.replace('_', " ")),
                            _ => format!("a pixel art game object {} in {}",
                                obj.obj_id, room_folder.replace('_', " ")),
                        };
                        result.lora_objects.push(LoraEntry {
                            image: format!("images/{}", lora_filename),
                            text: description,
                        });
                    }
                }
            }

            if !objects.is_empty() {
                result.log_lines.push(format!("  {}: {} object images", room_folder, objects.len()));
            }
        }
        Err(e) => result.log_lines.push(format!("  Warning: {} objects failed: {}", room_folder, e)),
    }

    // Build object metadata
    for om in &obj_metas {
        let obj_dir_name = match &om.name {
            Some(name) if !name.is_empty() => name.clone(),
            _ => format!("obj_{:05}", om.obj_id),
        };
        let image_files: Vec<String> = (0..om.num_states)
            .map(|s| format!("objects/{}/state_{:02}.png", sanitize_name(&obj_dir_name), s))
            .collect();
        obj_meta_list.push(ObjectMetadataJson {
            obj_id: om.obj_id,
            name: om.name.clone(),
            x: om.x,
            y: om.y,
            width: om.width,
            height: om.height,
            num_states: om.num_states,
            image_files,
        });
    }

    // Extract sounds
    let room_sounds = sound::extract_room_sounds(data, &entry.block, room_num, version);
    if !room_sounds.sounds.is_empty() {
        output.ensure_audio_room_dir(room_num)?;

        for (sound_idx, extracted_list) in &room_sounds.sounds {
            let mut sound_files = Vec::new();

            for extracted in extracted_list {
                let filename = format!("sound_{:03}_{}.{}",
                    sound_idx,
                    extracted.sound_type.label(),
                    extracted.sound_type.extension()
                );
                let path = output.room_audio_dir(room_num).join(&filename);
                std::fs::write(&path, &extracted.data)?;
                sound_files.push(filename.clone());
                result.sound_count += 1;
            }

            room_sound_refs.push(SoundRef {
                sound_id: *sound_idx,
                files: sound_files,
            });
        }

        result.log_lines.push(format!("  {}: {} sound files", room_folder, room_sounds.sounds.len()));
    }

    // Extract costumes (sprites)
    let mut sprite_meta_list: Vec<SpriteMetadataJson> = Vec::new();
    let room_palette = room::extract_palette(data, &room_block).ok();

    let costume_blocks: Vec<(usize, block::Block)> = match version {
        ScummVersion::V7 => costume::find_akos_blocks(data, &entry.block),
        _ => costume::find_cost_blocks(data, &entry.block),
    };

    for (costume_idx, costume_block) in &costume_blocks {
        let costume_info = match version {
            ScummVersion::V7 => costume::parse_akos(data, costume_block),
            _ => {
                let cost_data = &data[costume_block.data_offset()..costume_block.end_offset()];
                costume::parse_cost(cost_data, version)
            }
        };

        let costume_info = match costume_info {
            Ok(c) => c,
            Err(_) => continue,
        };

        if costume_info.frames.is_empty() {
            continue;
        }

        let costume_dir_name = format!("costume_{:03}", costume_idx);
        output.ensure_sprite_dir(room_num)?;
        let costume_path = output.room_sprite_dir(room_num).join(&costume_dir_name);
        std::fs::create_dir_all(costume_path.join("animations"))?;
        std::fs::create_dir_all(costume_path.join("frames"))?;

        // Save individual frames
        if let Some(ref pal) = room_palette {
            for (fi, frame) in costume_info.frames.iter().enumerate() {
                if frame.width == 0 || frame.height == 0 || frame.pixels.is_empty() {
                    continue;
                }
                let img = make_costume_image(&frame.pixels, frame.width, frame.height, &costume_info.palette, pal);
                let path = costume_path.join("frames").join(format!("frame_{:03}.png", fi));
                if let Err(e) = save_image(&img, &path) {
                    result.log_lines.push(format!("  Warning: costume frame save failed: {}", e));
                }
            }
        }

        // Save per-animation sprite sheets
        let mut anim_refs: Vec<SpriteAnimRef> = Vec::new();
        if let Some(ref pal) = room_palette {
            for anim in &costume_info.animations {
                let mut seen = HashSet::new();
                let mut all_frame_indices: Vec<usize> = Vec::new();
                for (_limb, frame_indices) in &anim.limb_frames {
                    for &fi in frame_indices {
                        if seen.insert(fi) {
                            all_frame_indices.push(fi);
                        }
                    }
                }

                let frame_refs: Vec<&costume::CostumeFrame> = all_frame_indices.iter()
                    .filter_map(|&fi| costume_info.frames.get(fi))
                    .filter(|f| f.width > 0 && f.height > 0 && !f.pixels.is_empty())
                    .collect();

                if frame_refs.is_empty() {
                    continue;
                }

                let (sw, sh, sheet_pixels) = costume::build_sprite_sheet(&frame_refs);
                if sw == 0 || sh == 0 {
                    continue;
                }

                let sheet_img = make_costume_image(&sheet_pixels, sw as u16, sh as u16, &costume_info.palette, pal);
                let filename = format!("{}.png", sanitize_name(&anim.name));
                let path = costume_path.join("animations").join(&filename);
                if let Err(e) = save_image(&sheet_img, &path) {
                    result.log_lines.push(format!("  Warning: costume sprite sheet save failed: {}", e));
                } else {
                    anim_refs.push(SpriteAnimRef {
                        name: anim.name.clone(),
                        file: format!("animations/{}", filename),
                        num_frames: frame_refs.len(),
                    });

                    let lora_filename = format!("{}_costume_{:03}_{}.png",
                        room_folder, costume_idx, sanitize_name(&anim.name));
                    let lora_path = output.lora_sprites_images_dir().join(&lora_filename);
                    let _ = save_image(&sheet_img, &lora_path);
                    result.lora_sprites.push(LoraEntry {
                        image: format!("images/{}", lora_filename),
                        text: format!("a pixel art character sprite sheet: {} animation",
                            anim.name.replace('_', " ")),
                    });
                }
            }
        }

        sprite_meta_list.push(SpriteMetadataJson {
            sprite_index: *costume_idx,
            num_frames: costume_info.frames.len(),
            num_animations: anim_refs.len(),
            animations: anim_refs,
            directory: format!("sprites/{}", costume_dir_name),
        });
        result.sprite_count += 1;
    }

    if !sprite_meta_list.is_empty() {
        result.log_lines.push(format!("  {}: {} sprites ({} total frames)",
            room_folder, sprite_meta_list.len(),
            sprite_meta_list.iter().map(|c| c.num_frames).sum::<usize>()));
    }

    // Write room metadata.json
    let room_name = output.room_folder_name(room_num);
    let raw_name = if room_name.starts_with("room_") { None } else { Some(room_name) };

    let room_meta = RoomMetadata {
        room_id: room_num,
        room_name: raw_name,
        width: header.width,
        height: header.height,
        palette_type: palette_type.to_string(),
        num_objects: obj_meta_list.len(),
        objects: obj_meta_list,
        num_sounds: room_sound_refs.len(),
        sounds: room_sound_refs,
        num_sprites: sprite_meta_list.len(),
        sprites: sprite_meta_list,
    };

    let meta_path = output.room_dir(room_num).join("metadata.json");
    let json = serde_json::to_string_pretty(&room_meta)?;
    std::fs::write(&meta_path, json)?;

    Ok(result)
}

/// Extract V3/V4 games (individual .LFL files)
fn extract_v3(game: &version::GameInfo, output: &OutputManager, prefix: &str, summary: &mut ExtractionSummary, progress: Option<&ProgressBar>) -> Result<()> {
    let index_data = std::fs::read(&game.index_path)
        .context("Failed to read index file")?;

    let game_index = index::parse_index_v3(&index_data)
        .context("Failed to parse V3 index")?;
    summary.log_lines.push(format!("{}: found {} rooms in V3 index", prefix, game_index.room_directory.len()));

    let rooms = resource::parse_v3_rooms(&game.data_path, &game_index.room_directory)?;
    summary.log_lines.push(format!("{}: loaded {} room files", prefix, rooms.len()));

    output.ensure_lora_dirs()?;
    let mut lora_backgrounds: Vec<LoraEntry> = Vec::new();
    let mut lora_objects: Vec<LoraEntry> = Vec::new();

    if let Some(pb) = progress {
        pb.set_message(format!("{} rooms", rooms.len()));
    }

    for (room_num, room_data) in &rooms {
        let room_id = *room_num as u16;
        output.ensure_room_dirs(room_id)?;

        let room_folder = output.room_folder_name(room_id);

        // Parse V3 room blocks: outer RO block, then inner child blocks
        let ro_block = match block::parse_block_v3(room_data, 0) {
            Ok(b) if b.tag_v3() == b"RO" => b,
            _ => {
                summary.log_lines.push(format!("{}: room {:03}: no RO block found", prefix, room_id));
                continue;
            }
        };
        let inner_blocks = block::iter_children_v3(room_data, ro_block.data_offset_v3(), ro_block.end_offset());

        // Extract header (HD block)
        let (width, height) = if let Some(hd) = inner_blocks.iter().find(|b| b.tag_v3() == b"HD") {
            let d = &room_data[hd.data_offset_v3()..hd.end_offset()];
            if d.len() >= 4 {
                (u16::from_le_bytes([d[0], d[1]]), u16::from_le_bytes([d[2], d[3]]))
            } else {
                (0, 0)
            }
        } else {
            (0, 0)
        };

        // Extract palette (PA block)
        let palette = if let Some(pa) = inner_blocks.iter().find(|b| b.tag_v3() == b"PA") {
            extract_v3_palette(room_data, pa)
        } else {
            room::Palette { colors: {
                let mut c = [(0u8, 0u8, 0u8); 256];
                for i in 0..256 { c[i] = (i as u8, i as u8, i as u8); }
                c
            }}
        };

        // Extract background (BM block)
        if width > 0 && height > 0 {
            if let Some(bm) = inner_blocks.iter().find(|b| b.tag_v3() == b"BM") {
                match decode_v3_background(room_data, bm, width as usize, height as usize, &palette) {
                    Ok(img) => {
                        let path = output.room_image_dir(room_id).join("background.png");
                        match save_image(&img, &path) {
                            Ok(_) => {
                                summary.log_lines.push(format!("{}: {}: background {}x{}", prefix, room_folder, width, height));
                                summary.backgrounds += 1;

                                let lora_filename = format!("{}.png", room_folder);
                                let lora_path = output.lora_backgrounds_images_dir().join(&lora_filename);
                                let _ = save_image(&img, &lora_path);
                                lora_backgrounds.push(LoraEntry {
                                    image: format!("images/{}", lora_filename),
                                    text: format!("a pixel art scene of {}", room_folder.replace('_', " ")),
                                });
                            }
                            Err(e) => summary.log_lines.push(format!("{}: {}: background save failed: {}", prefix, room_folder, e)),
                        }
                    }
                    Err(e) => summary.log_lines.push(format!("{}: {}: background decode failed: {}", prefix, room_folder, e)),
                }
            }
        }

        // Extract V3 object images (OI blocks with OC metadata)
        let oi_blocks: Vec<_> = inner_blocks.iter().filter(|b| b.tag_v3() == b"OI").collect();
        let oc_blocks: Vec<_> = inner_blocks.iter().filter(|b| b.tag_v3() == b"OC").collect();
        let mut obj_meta_list: Vec<ObjectMetadataJson> = Vec::new();

        for (idx, oi) in oi_blocks.iter().enumerate() {
            let oc = oc_blocks.get(idx);

            // Parse object metadata from OC block
            let (obj_id, obj_x, obj_y, obj_w, obj_h, obj_name, num_states) = if let Some(oc) = oc {
                parse_v3_object_header(room_data, oc)
            } else {
                (idx as u16, 0, 0, 0, 0, None, 1)
            };

            let oi_data = &room_data[oi.data_offset_v3()..oi.end_offset()];

            // Object image dimensions: use OC dimensions, or fall back to guessing from data
            let ow = if obj_w > 0 { obj_w as usize } else { continue };
            let oh = if obj_h > 0 { obj_h as usize } else { continue };

            let obj_dir_name = match &obj_name {
                Some(name) if !name.is_empty() => sanitize_name(name),
                _ => format!("obj_{:05}", obj_id),
            };

            match decode_v3_image_strips(oi_data, ow, oh) {
                Ok(pixels) => {
                    let img = room::DecodedImage { width: ow as u16, height: oh as u16, pixels, palette };

                    output.ensure_room_object_dir(room_id, &obj_dir_name)?;
                    let path = output.room_object_dir(room_id, &obj_dir_name).join("state_00.png");
                    if save_image(&img, &path).is_ok() {
                        summary.objects += 1;

                        let lora_filename = format!("{}_{}_state_00.png", room_folder, &obj_dir_name);
                        let lora_path = output.lora_objects_images_dir().join(&lora_filename);
                        let _ = save_image(&img, &lora_path);
                        let description = match &obj_name {
                            Some(name) if !name.is_empty() =>
                                format!("a pixel art object: {}", name.replace('_', " ")),
                            _ => format!("a pixel art game object {} in {}", obj_id, room_folder.replace('_', " ")),
                        };
                        lora_objects.push(LoraEntry {
                            image: format!("images/{}", lora_filename),
                            text: description,
                        });
                    }
                }
                Err(_) => {} // silently skip failed object decodes
            }

            let image_files = vec![format!("objects/{}/state_00.png", obj_dir_name)];
            obj_meta_list.push(ObjectMetadataJson {
                obj_id,
                name: obj_name.clone(),
                x: obj_x,
                y: obj_y,
                width: obj_w,
                height: obj_h,
                num_states,
                image_files,
            });
        }

        if !oi_blocks.is_empty() {
            summary.log_lines.push(format!("{}: {}: {} object images", prefix, room_folder, oi_blocks.len()));
        }

        // Write room metadata.json
        let room_meta = RoomMetadata {
            room_id,
            room_name: None,
            width,
            height,
            palette_type: "v3".to_string(),
            num_objects: obj_meta_list.len(),
            objects: obj_meta_list,
            num_sounds: 0,
            sounds: Vec::new(),
            num_sprites: 0,
            sprites: Vec::new(),
        };

        let meta_path = output.room_dir(room_id).join("metadata.json");
        let json = serde_json::to_string_pretty(&room_meta)?;
        std::fs::write(&meta_path, json)?;

        if let Some(pb) = progress { pb.tick(); }
    }

    summary.log_lines.push(format!("{}: processed {} V3 rooms", prefix, rooms.len()));

    // Write lora_training JSONL metadata files
    if !lora_backgrounds.is_empty() {
        write_jsonl(&lora_backgrounds, &output.lora_backgrounds_dir().join("metadata.jsonl"))?;
        summary.log_lines.push(format!("{}: lora training: {} background images", prefix, lora_backgrounds.len()));
    }
    if !lora_objects.is_empty() {
        write_jsonl(&lora_objects, &output.lora_objects_dir().join("metadata.jsonl"))?;
        summary.log_lines.push(format!("{}: lora training: {} object images", prefix, lora_objects.len()));
    }

    Ok(())
}

/// Extract palette from a V3 PA block.
/// PA block data has a 2-byte prefix (format indicator) followed by palette entries.
fn extract_v3_palette(data: &[u8], pa: &block::Block) -> room::Palette {
    let d = &data[pa.data_offset_v3()..pa.end_offset()];
    let mut colors = [(0u8, 0u8, 0u8); 256];

    // Skip 2-byte prefix
    let pal_data = if d.len() > 2 { &d[2..] } else { d };

    if pal_data.len() >= 768 {
        // VGA 256-color palette
        for i in 0..256 {
            colors[i] = (pal_data[i * 3], pal_data[i * 3 + 1], pal_data[i * 3 + 2]);
        }
    } else if pal_data.len() >= 48 {
        // EGA 16-color palette
        for i in 0..16.min(pal_data.len() / 3) {
            colors[i] = (pal_data[i * 3], pal_data[i * 3 + 1], pal_data[i * 3 + 2]);
        }
    } else {
        // Fallback grayscale
        for i in 0..256 {
            colors[i] = (i as u8, i as u8, i as u8);
        }
    }

    room::Palette { colors }
}

/// Parse V3 object header from OC block. Returns (obj_id, x, y, width, height, name, num_states).
fn parse_v3_object_header(data: &[u8], oc: &block::Block) -> (u16, u16, u16, u16, u16, Option<String>, u8) {
    let d = &data[oc.data_offset_v3()..oc.end_offset()];
    if d.len() < 13 {
        return (0, 0, 0, 0, 0, None, 1);
    }

    // V3 OC header: obj_id(2), unk(1), x(1), y_and_parent(1), width(1), unk(1), height_and_actor_dir(1)
    let obj_id = u16::from_le_bytes([d[0], d[1]]);
    let x = d[3] as u16 * 8;
    let y = (d[4] & 0x7F) as u16 * 8;
    let width = d[5] as u16 * 8;
    let height = (d[7] & 0xF8) as u16;

    // Object name: scan for null-terminated string after the header area
    // In V3, the name is embedded in the OC block after script data — try to find it
    let name = extract_v3_obj_name(d);

    (obj_id, x, y, width, height, name, 1)
}

/// Try to extract the object name from a V3 OC block.
/// The name is typically after the verb table, null-terminated.
fn extract_v3_obj_name(oc_data: &[u8]) -> Option<String> {
    // V3 OC: after fixed header and verb entries, there's a name string
    // The verb table starts at offset 13 or 14 and ends with 0x00
    if oc_data.len() < 15 {
        return None;
    }
    // Scan for the name: look for printable ASCII string followed by null
    let mut pos = 13;
    // Skip verb table entries (each is 3 bytes: verb_id, offset_lo, offset_hi) until verb_id == 0
    while pos < oc_data.len() {
        if oc_data[pos] == 0 {
            pos += 1;
            break;
        }
        pos += 3;
    }
    if pos >= oc_data.len() {
        return None;
    }
    // Now we should be at the name
    let name_start = pos;
    while pos < oc_data.len() && oc_data[pos] != 0 {
        pos += 1;
    }
    if pos > name_start {
        String::from_utf8(oc_data[name_start..pos].to_vec()).ok()
    } else {
        None
    }
}

/// Decode V3 background image from a BM block.
fn decode_v3_background(data: &[u8], bm: &block::Block, width: usize, height: usize, palette: &room::Palette) -> Result<room::DecodedImage> {
    let bm_data = &data[bm.data_offset_v3()..bm.end_offset()];
    let pixels = decode_v3_image_strips(bm_data, width, height)?;
    Ok(room::DecodedImage {
        width: width as u16,
        height: height as u16,
        pixels,
        palette: *palette,
    })
}

/// Decode V3 image strip data. The data starts with a u32 LE offset table.
fn decode_v3_image_strips(strip_data: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let strip_count = width / 8;
    if strip_count == 0 {
        anyhow::bail!("V3 image width {} too small for strips", width);
    }

    // V3 strip offset table: strip_count * u32 LE offsets relative to strip_data start
    if strip_data.len() < strip_count * 4 {
        anyhow::bail!("V3 strip data too small for offset table");
    }

    let mut pixels = vec![0u8; width * height];

    for strip_idx in 0..strip_count {
        let off = u32::from_le_bytes([
            strip_data[strip_idx * 4],
            strip_data[strip_idx * 4 + 1],
            strip_data[strip_idx * 4 + 2],
            strip_data[strip_idx * 4 + 3],
        ]) as usize;

        if off >= strip_data.len() {
            continue;
        }

        let sdata = &strip_data[off..];
        match super::image_decode::decode_strip_v3(sdata, height) {
            Ok(strip_pixels) if strip_pixels.len() == 8 * height => {
                let x_start = strip_idx * 8;
                for y in 0..height {
                    pixels[y * width + x_start..y * width + x_start + 8]
                        .copy_from_slice(&strip_pixels[y * 8..y * 8 + 8]);
                }
            }
            Ok(_) | Err(_) => {} // skip malformed or failed strips
        }
    }

    Ok(pixels)
}

/// Extract speech from MONSTER.SOU/sof/so3/sog file. Returns number of speech files extracted.
fn extract_speech(sou_path: &std::path::Path, output: &OutputManager, prefix: &str, log: &mut Vec<String>) -> Result<usize> {
    output.ensure_speech_dir()?;

    match monster_sou::parse_monster_file(sou_path) {
        Ok(entries) => {
            // Write speech files in parallel
            let speech_meta: Vec<SpeechMetadataJson> = entries.par_iter()
                .filter_map(|entry| {
                    let filename = format!("speech_{:05}.{}", entry.index, entry.format.extension());
                    let path = output.speech_dir().join(&filename);
                    std::fs::write(&path, &entry.audio_data).ok()?;

                    Some(SpeechMetadataJson {
                        index: entry.index,
                        format: entry.format.label().to_string(),
                        file: filename,
                        sample_rate: entry.sample_rate,
                    })
                })
                .collect();

            let count = speech_meta.len();
            log.push(format!("{}: extracted {} speech entries", prefix, count));

            // Write speech metadata
            let meta_path = output.speech_dir().join("metadata.json");
            let json = serde_json::to_string_pretty(&speech_meta)?;
            std::fs::write(&meta_path, json)?;

            Ok(count)
        }
        Err(e) => {
            log.push(format!("{}: warning: failed to parse speech file: {}", prefix, e));
            Ok(0)
        }
    }
}

/// Remap pixel indices through a costume palette.
/// Index 0 is transparent (kept as 0/black).
fn remap_costume_pixels(input: &[u8], costume_palette: &[u8]) -> Vec<u8> {
    input.iter().map(|&idx| {
        if idx == 0 {
            0
        } else if (idx as usize) < costume_palette.len() {
            costume_palette[idx as usize]
        } else {
            idx
        }
    }).collect()
}

fn make_costume_image(
    pixels: &[u8],
    width: u16,
    height: u16,
    costume_palette: &[u8],
    room_palette: &room::Palette,
) -> room::DecodedImage {
    room::DecodedImage {
        width,
        height,
        pixels: remap_costume_pixels(pixels, costume_palette),
        palette: *room_palette,
    }
}
