# Adventure Game Asset Extractor

A Rust CLI tool that extracts assets (images, audio, sprites) from classic adventure game files. Supports both LucasArts SCUMM engine games (v3-v7) and Sierra SCI engine games (SCI0 through SCI2.1).

## Usage

The tool has two subcommands: `extract` and `analyze`.

### Extract

Extract assets from game files. Accepts one or more game directory paths:
```
cargo run -- extract <GAME_DIR>... [-o output_dir] [--analyze --model <MODEL>]
```

Point at a parent folder and all games are detected and extracted automatically:
```
cargo run -- extract "../games" -o ./output
```

Extract multiple specific games:
```
cargo run -- extract "../games/scumm/dott" "../games/scumm/MonkeyIsland1" -o ./output
```

The tool recursively scans subdirectories for known game formats. When a game is detected in a folder, it does not search deeper within that folder. All output is prefixed with the game name for clarity, and a summary is printed after completion.

#### Extract + Analyze

Run analysis immediately after extraction using `--analyze`:
```
cargo run -- extract "../games" -o ./output --analyze --model gamma-3
```

Additional analyze options: `--endpoint <URL>` (default: http://127.0.0.1:1234), `--parallel <N>` (default: 4).

### Analyze

Generate image descriptions for LoRA training datasets using a local LLM (via LM Studio or any OpenAI-compatible API):
```
cargo run -- analyze <OUTPUT_DIR> --model <MODEL> [--endpoint <URL>] [--parallel <N>] [--output-mode <MODE>] [--force]
```

The analyze command auto-discovers all `lora_training` subdatasets under the output directory and labels images that don't yet have descriptions.

### Examples
```bash
# Single game extraction
cargo run -- extract "../games/scumm/MonkeyIsland1" -o ./output

# Multiple specific games
cargo run -- extract "../games/scumm/dott" "../games/scumm/MonkeyIsland1" -o ./output

# All games in a folder tree (SCUMM + SCI)
cargo run -- extract "../games" -o ./output

# Extract and immediately analyze with LLM
cargo run -- extract "../games" -o ./output --analyze --model gamma-3

# Analyze extracted images (standalone)
cargo run -- analyze ./output --model gamma-3

# Force re-label all images, write directly to metadata.jsonl
cargo run -- analyze ./output --model gamma-3 --output-mode update --force
```

## Supported Games

### SCUMM Engine (LucasArts)

| Game | SCUMM Version | Files |
|------|--------------|-------|
| Indiana Jones and the Last Crusade | V3 | 00.LFL / NN.LFL (backgrounds + objects) |
| The Secret of Monkey Island | V5 | .000/.001 |
| Monkey Island 2: LeChuck's Revenge | V5 | .000/.001 |
| Indiana Jones and the Fate of Atlantis | V5 | .000/.001 + MONSTER.SOU |
| Day of the Tentacle | V6 | .000/.001 + monster.sof |
| Sam & Max Hit the Road | V6 | .000/.001 + MONSTER.SOU |
| Full Throttle | V7 | .LA0/.LA1 + monster.sof |

### SCI Engine (Sierra)

| Game | SCI Version | Assets Extracted |
|------|------------|-----------------|
| King's Quest 1 VGA Remake | SCI0 | Pics, Views, Sounds |
| King's Quest 4 (SCI version) | SCI0 | Pics, Views, Sounds |
| King's Quest 5 | SCI1 Late | Pics, Views, Sounds |
| King's Quest 6 | SCI1.1 | Pics, Views, Sounds |
| Space Quest 1 VGA Remake | SCI1 Late | Pics, Views, Sounds |
| Space Quest 3 | SCI0 | Pics, Views, Sounds |
| Space Quest 4 | SCI1.1 | Pics, Views, Sounds |
| Space Quest 5 | SCI1.1 | Pics, Views, Sounds |
| Leisure Suit Larry 1 | SCI1 Middle | Pics, Views, Sounds |
| Leisure Suit Larry 2 | SCI0 | Pics, Views, Sounds |
| Leisure Suit Larry 3 | SCI0 | Pics, Views, Sounds |
| Leisure Suit Larry 5 | SCI1 Late | Pics, Views, Sounds |
| Police Quest 1 VGA Remake | SCI1 Late | Pics, Views, Sounds |
| Police Quest 2 | SCI0 | Pics, Views, Sounds |
| Police Quest 3 | SCI1 Late | Pics, Views, Sounds |
| Quest for Glory 1 (Hero's Quest) | SCI0 | Pics, Views, Sounds |
| Leisure Suit Larry 6 | SCI2.1 | Resource loading only* |
| Leisure Suit Larry 7 | SCI2.1 | Resource loading only* |
| Space Quest 6 | SCI2.1 | Resource loading only* |

\* SCI2.1 games load and decompress resources but use a different bitmap format for pics/views that is not yet rendered.

**Note:** AGI-era Sierra games (pre-SCI) are not supported: KQ1-3 original, SQ1-2 original, PQ1 original.

## Output Structure

### SCUMM Games
```
output/<game_name>/
  assets/
    rooms/<room_name>/
      images/
        background.png
        objects/<obj_name>/state_NN.png
      audio/sound_NNN_type.ext
      costumes/costume_NNN/
        frames/frame_NNN.png
        animations/<anim_name>.png
      metadata.json
    speech/
      speech_NNNNN.wav
      metadata.json
  lora_training/
    backgrounds/
      metadata.jsonl
      images/<room_name>.png
    objects/
      metadata.jsonl
      images/<room>_<obj_name>_state_NN.png
    costumes/
      metadata.jsonl
      images/<room>_costume_NNN_<anim_name>.png
```

### SCI Games
```
output/<game_name>/
  assets/
    pics/pic_NNN/
      background.png
    views/view_NNN/
      loop_NN/cel_NN.png
      loop_NN_sheet.png
    audio/sound_NNN.mid
  lora_training/
    backgrounds/
      metadata.jsonl
      images/pic_NNN.png
    sprites/
      metadata.jsonl
      images/view_NNN_loop_NN.png
```

The `assets/` folder contains the full hierarchical extraction. The `lora_training/` folder contains flat copies of all images with `metadata.jsonl` files for training image diffusion LoRA models. Each line in a JSONL file is a JSON object with `image` (relative path) and `text` (description) fields:

```jsonl
{"image":"images/beach.png","text":"a pixel art scene of beach"}
{"image":"images/bar_door_state_00.png","text":"a pixel art object: door"}
```

## Performance

The tool uses parallel processing (rayon) at multiple levels:
- **Multi-game**: When extracting multiple games, each game is processed in parallel
- **Room extraction**: Within each game, rooms are extracted in parallel across all CPU cores
- **Speech extraction**: MONSTER.SOU speech entries are written in parallel

## Module Architecture

```
src/
├── main.rs              CLI entry, subcommands, game discovery, dispatch
├── engine.rs            Engine trait (for multi-engine support)
├── analyze/
│   ├── mod.rs           Analyze orchestration, subdataset discovery
│   ├── api.rs           LM Studio API client (OpenAI-compatible)
│   ├── dataset.rs       Dataset I/O (load images, read/write metadata.jsonl)
│   └── prompt.rs        LLM prompt definitions
├── common/
│   ├── bitstream.rs     LSB-first bit reader
│   ├── audio_convert.rs VOC/WAV audio conversion
│   ├── midi_extract.rs  MIDI format utilities
│   ├── decrypt.rs       XOR decryption
│   ├── output.rs        Output directory management, image saving
│   ├── metadata.rs      JSON/JSONL metadata structs
│   └── progress.rs      Progress bar styles and helpers
├── scumm/
│   ├── mod.rs           ScummEngine struct + Engine impl
│   ├── extract.rs       Extraction orchestration (parallel room processing)
│   ├── version.rs       Game detection, ScummVersion enum
│   ├── block.rs         Block parser and tree traversal
│   ├── index.rs         Index file (.000) parsing
│   ├── resource.rs      Data file (.001) navigation
│   ├── room.rs          Room extraction (header, palette, images)
│   ├── image_decode.rs  Strip decompression codecs
│   ├── sound.rs         Sound extraction
│   ├── monster_sou.rs   MONSTER.SOU speech extraction
│   └── costume.rs       Costume parsing (COST V5/V6, AKOS V7)
└── sci/
    ├── mod.rs           SciEngine struct + Engine impl + detect()
    ├── extract.rs       Extraction orchestration
    ├── version.rs       Game/version detection (SCI0 through SCI2.1)
    ├── resource_map.rs  Resource map parsing (all SCI map formats)
    ├── resource_volume.rs Volume header parsing (SCI0/SCI1/SCI1.1/SCI2)
    ├── resource.rs      ResourceManager (load + decompress)
    ├── decompress.rs    Huffman, DCL Implode, LZS/STACpack decompression
    ├── palette.rs       Palette parsing (EGA default, SCI0, SCI1.1 formats)
    ├── picture.rs       Picture rendering (vector SCI0/SCI1, bitmap SCI1.1)
    ├── view.rs          View/sprite extraction (SCI0 EGA, SCI1 VGA, SCI1.1 VGA)
    └── sound.rs         Sound/MIDI extraction
```

### Adding a New Engine

To add support for another game engine:

1. Create a new module directory (e.g., `src/agi/`)
2. Implement the `Engine` trait from `engine.rs`
3. Add a `detect()` method that identifies game files
4. Register the engine in `main.rs::try_detect()`

Shared utilities in `common/` (audio conversion, bitstream reading, output management) can be reused across engines.

## Dependencies

- **image** -- PNG encoding
- **clap** -- CLI argument parsing
- **anyhow** -- Error handling
- **serde/serde_json** -- JSON/JSONL serialization
- **rayon** -- Parallel processing
- **tokio** -- Async runtime (for analyze command)
- **reqwest** -- HTTP client (for LLM API)
- **base64** -- Base64 encoding (for image payloads)
- **indicatif** -- Progress bars and spinners
- **console** -- Terminal styling utilities
