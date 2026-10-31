use std::fs;
use std::path::Path;

use fern::colors::{Color, ColoredLevelConfig};
use log::{LevelFilter};

pub fn setup_logger() -> Result<(), fern::InitError> {
    // Configure the colors for your console output
    let colors = ColoredLevelConfig::new()
        .info(Color::Green)
        .debug(Color::Blue)
        .warn(Color::Yellow)
        .error(Color::Red)
        .trace(Color::Magenta);

    // 1. Console Dispatcher (Colored)
    let stdout_log = fern::Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!(
                "[{timestamp}] [{level}] {message}",
                timestamp = humantime::format_rfc3339(std::time::SystemTime::now()),
                level = colors.color(record.level()),
                message = message
            ))
        })
        .chain(std::io::stdout());

    // 2. File Dispatcher (Uncolored)
    // If an environment variable has been passed that has a file, make sure that file has a
    // path that exists.
    let mut file_log_option = None;
    if let Ok(log_file_path) = std::env::var("LOG_FILE") {
        if let Some(parent) = Path::new(&log_file_path)
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            if !parent.exists() {
                fs::create_dir_all(parent)?;
            }
        }

        file_log_option = Some(fern::Dispatch::new()
            .format(|out, message, record| {
                out.finish(format_args!(
                    "[{timestamp}] [{level}] {message}",
                    timestamp = humantime::format_rfc3339(std::time::SystemTime::now()),
                    level = record.level(), // No color wrapping here
                    message = message
                ))
            })
            .chain(fern::log_file(log_file_path)?));
    }

    // 3. Base Dispatcher (Combines both sinks)
    let mut dispatch = fern::Dispatch::new()
        // Set your global minimum log level
        .level(LevelFilter::Debug)
        .chain(stdout_log);

    if file_log_option.is_some() {
        dispatch = dispatch.chain(file_log_option.unwrap());
    }

    dispatch.apply()?;

    Ok(())
}