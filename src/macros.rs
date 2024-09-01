#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {
        print!("\r{}", termion::clear::CurrentLine);
        print!($($arg)*);
        std::io::Write::flush(&mut std::io::stdout());
    }
}

#[macro_export]
macro_rules! clear_log {
    () => {
        print!("\r{}", termion::clear::CurrentLine);
    }
}
