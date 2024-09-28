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

#[macro_export]
macro_rules! warn {
    (@basic $($arg:tt)*) => {
        print!(
            "{}{}warning: {}",
            termion::color::Fg(termion::color::Yellow),
            termion::style::Bold,
            termion::color::Fg(termion::color::Reset),
        );
        print!($($arg)*);
        print!("\n\n");
        std::io::Write::flush(&mut std::io::stdout());
    };
    ($loc:expr, $($arg:tt)*) => {
        print!(
            "{}{}warning: {}",
            termion::color::Fg(termion::color::Yellow),
            termion::style::Bold,
            termion::color::Fg(termion::color::Reset),
        );
        print!($($arg)*);
        print!(
            "\n  {}-->{}{} {}:{}:{}\n",
            termion::color::Fg(termion::color::Blue),
            termion::color::Fg(termion::color::Reset),
            termion::style::Reset,
            $loc.filename,
            $loc.line,
            $loc.start,
        );
        std::io::Write::flush(&mut std::io::stdout());
    };
    (@image $file:expr, $($arg:tt)*) => {
        print!(
            "{}{}warning: {}",
            termion::color::Fg(termion::color::Yellow),
            termion::style::Bold,
            termion::color::Fg(termion::color::Reset),
        );
        print!($($arg)*);
        print!(
            "\n  {}-->{}{} {}\n",
            termion::color::Fg(termion::color::Blue),
            termion::color::Fg(termion::color::Reset),
            termion::style::Reset,
            $file,
        );
        std::io::Write::flush(&mut std::io::stdout());
    };
}

#[macro_export]
macro_rules! error {
    (@panic $($arg:tt)*) => {
        print!(
            "{}{}error: {}",
            termion::color::Fg(termion::color::Red),
            termion::style::Bold,
            termion::color::Fg(termion::color::Reset),
        );
        print!($($arg)*);
        print!("\n\n");
        std::io::Write::flush(&mut std::io::stdout());
        panic!();
    };
    (@noloc $($arg:tt)*) => {
        return Err(error!(@create @noloc $($arg)*));
    };
    (@create @noloc $($arg:tt)*) => {
        {
            $crate::reader::error::ParseError::new(
                $crate::reader::error::ParseErrorKind::InvalidParameter,
                $crate::reader::target::FileLoc::default(),
                Some(format!($($arg)*)),
            )
        }
    };
    ($loc:expr, $($arg:tt)*) => {
        return Err(error!(@create $loc, $($arg)*));
    };
    (@create $loc:expr, $($arg:tt)*) => {
        {
            $crate::reader::error::ParseError::new(
                $crate::reader::error::ParseErrorKind::InvalidParameter,
                $loc.clone(),
                Some(format!($($arg)*)),
            )
        }
    };
    (@file $file:expr, $($arg:tt)*) => {
        return Err(error!(@create @file $file, $($arg)*))
    };
    (@create @file $file:expr, $($arg:tt)*) => {
        {
            $crate::reader::error::ParseError::new(
                $crate::reader::error::ParseErrorKind::InvalidParameter,
                $crate::reader::target::FileLoc::from_file($file),
                Some(format!($($arg)*)),
            )
        }
    };
    // (@basic $($arg:tt)*) => {
    //     print!(
    //         "{}{}error: {}", 
    //         termion::color::Fg(termion::color::Red),
    //         termion::style::Bold,
    //         termion::color::Fg(termion::color::Reset),
    //     );
    //     print!($($arg)*);
    //     print!("\n\n");
    //     std::io::Write::flush(&mut std::io::stdout());
    //     panic!();
    // };
    // ($loc:expr, $($arg:tt)*) => {
    //     print!(
    //         "{}{}error: {}", 
    //         termion::color::Fg(termion::color::Red),
    //         termion::style::Bold,
    //         termion::color::Fg(termion::color::Reset),
    //     );
    //     print!($($arg)*);
    //     print!(
    //         "\n  {}-->{}{} {}:{}:{}\n",
    //         termion::color::Fg(termion::color::Blue),
    //         termion::color::Fg(termion::color::Reset),
    //         termion::style::Reset,
    //         $loc.filename,
    //         $loc.line,
    //         $loc.start,
    //     );
    //     std::io::Write::flush(&mut std::io::stdout());
    //     panic!();
    // };
    // (@image $file:expr, $($arg:tt)*) => {
    //     print!(
    //         "{}{}error: {}", 
    //         termion::color::Fg(termion::color::Red),
    //         termion::style::Bold,
    //         termion::color::Fg(termion::color::Reset),
    //     );
    //     print!($($arg)*);
    //     print!(
    //         "\n  {}-->{}{} {}\n",
    //         termion::color::Fg(termion::color::Blue),
    //         termion::color::Fg(termion::color::Reset),
    //         termion::style::Reset,
    //         $file,
    //     );
    //     std::io::Write::flush(&mut std::io::stdout());
    //     panic!();
    // };
}

#[macro_export]
macro_rules! new_syntax_err {
    // ($kind:ident, $loc:expr, $msg:expr) => {
    //     $crate::reader::error::ParseError::new($crate::reader::error::ParseErrorKind::$kind, $loc, Some($msg))
    // };
    ($kind:ident, $loc:expr, $($msg:tt)*) => {
        $crate::reader::error::SyntaxError::new($crate::reader::error::SyntaxErrorKind::$kind, $loc, Some(format!($($msg)*)))
    };
    ($kind:ident, $loc:expr) => {
        $crate::reader::error::SyntaxError::new($crate::reader::error::SyntaxErrorKind::$kind, $loc, None)
    };
    ($kind:expr, $loc:expr) => {
        $crate::reader::error::SyntaxError::new($kind, $loc, None)
    };
}
