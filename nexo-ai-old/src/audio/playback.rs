use std::num::NonZero;

use rodio::Player;
use rodio::buffer::SamplesBuffer;
use rodio::stream::{DeviceSinkBuilder, MixerDeviceSink};

use super::AudioBuffer;

/// Handle returned by [`play_async`] to control non-blocking playback.
pub struct PlaybackHandle {
    player: Player,
    // Keep the device sink alive for the duration of playback.
    _sink: MixerDeviceSink,
}

impl PlaybackHandle {
    /// Block the current thread until playback finishes.
    pub fn wait(self) {
        self.player.sleep_until_end();
    }

    /// Stop playback immediately.
    pub fn stop(self) {
        self.player.stop();
    }
}

/// Create a `SamplesBuffer` from an `AudioBuffer`.
fn make_source(buffer: &AudioBuffer) -> anyhow::Result<SamplesBuffer> {
    let channels = NonZero::new(buffer.channels)
        .ok_or_else(|| anyhow::anyhow!("channel count must be non-zero"))?;
    let sample_rate = NonZero::new(buffer.sample_rate)
        .ok_or_else(|| anyhow::anyhow!("sample rate must be non-zero"))?;
    Ok(SamplesBuffer::new(
        channels,
        sample_rate,
        buffer.samples.clone(),
    ))
}

/// Open default sink and start playing the buffer. Returns the player and sink.
fn start_playback(buffer: &AudioBuffer) -> anyhow::Result<(Player, MixerDeviceSink)> {
    let sink = DeviceSinkBuilder::open_default_sink()
        .map_err(|e| anyhow::anyhow!("failed to open default audio output: {e}"))?;
    let player = Player::connect_new(sink.mixer());
    player.append(make_source(buffer)?);
    Ok((player, sink))
}

/// Play an `AudioBuffer` to the default audio output, blocking until complete.
pub fn play(buffer: &AudioBuffer) -> anyhow::Result<()> {
    let (_player, _sink) = start_playback(buffer)?;

    // Wait for playback to finish
    std::thread::sleep(std::time::Duration::from_secs_f64(
        buffer.duration_secs() + 0.1,
    ));

    Ok(())
}

/// Start playback of an `AudioBuffer` on the default audio output and return
/// immediately with a [`PlaybackHandle`].
///
/// The caller must hold onto the handle -- dropping it will stop playback.
pub fn play_async(buffer: &AudioBuffer) -> anyhow::Result<PlaybackHandle> {
    let (player, sink) = start_playback(buffer)?;

    Ok(PlaybackHandle {
        player,
        _sink: sink,
    })
}
