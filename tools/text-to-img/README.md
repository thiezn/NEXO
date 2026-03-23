# Text to Image Generator

Generates images from given text imput using local LLMs and image generation models.

Both a Rust library (importable from other crates) and a CLI tool.


## CLI Usage

```sh
# From inline text
cargo run -p text_to_img -- "A sunset over the ocean with dramatic clouds"

# From a file
cargo run -p text_to_img -- prompt.txt

# With options
cargo run -p text_to_img -- "Mountain landscape" \
  --image-model flux.2 \
  --width 1024 --height 768 \
  -n 2 \
  --lora my-style-lora
```

## Library Usage

```rust
let config = text_to_img::GenerationConfig {
    image_model: "flux.2".to_string(),
    ..Default::default()
};

let result = text_to_img::generate_images(
    &["A chapter about mountains and solitude.".to_string()],
    &config,
).await?;

println!("Prompt: {}", result.prompt_used);
for image in &result.images {
    println!("Image {}: {} bytes", image.index, image.base64_data.len());
}
```
