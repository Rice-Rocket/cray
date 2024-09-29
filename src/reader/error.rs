use std::{io, num::{ParseFloatError, ParseIntError}, str::ParseBoolError};

use thiserror::Error;

use crate::Wrap;

use super::target::FileLoc;

pub type ParseResult<T> = std::result::Result<T, ParseError>;

#[derive(PartialEq, PartialOrd)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub loc: FileLoc,
    pub msg: Option<String>,
}

impl ParseError {
    #[inline]
    pub fn new(kind: ParseErrorKind, loc: FileLoc, msg: Option<String>) -> ParseError {
        ParseError { kind, loc, msg }
    }

    #[inline]
    pub fn with_loc(self, loc: FileLoc) -> ParseError {
        ParseError {
            kind: self.kind,
            loc,
            msg: self.msg,
        }
    }

    pub fn format(&self, s: &str) -> String {
        use termion::{color, style};

        let line_digits = self.loc.line.checked_ilog10().unwrap_or(0) + 1;
        let indent = " ".repeat(line_digits as usize);

        let mut res = format!(
            // error: [reason]
            //     --> [filename]:[line]:[col]
            //      |
            "{}{}error: {}{}\n{indent}{}-->{}{} {}:{}:{}\n{indent} {}{}|\n",
            color::Fg(color::Red),
            style::Bold,
            color::Fg(color::Reset),
            self.kind,
            color::Fg(color::Blue),
            color::Fg(color::Reset),
            style::Reset,
            self.loc.filename,
            self.loc.line,
            self.loc.start,
            color::Fg(color::Blue),
            style::Bold,
        );

        let line_contents = s.lines().nth(self.loc.line as usize).unwrap_or("");

        res += &format!(
            "{}{}{} |{}{} {}\n",
            color::Fg(color::Blue),
            style::Bold,
            self.loc.line,
            color::Fg(color::Reset),
            style::Reset,
            line_contents,
        );

        let indent_offset = " ".repeat(self.loc.start as usize);
        let arrows = "^".repeat(self.loc.len as usize);

        res += &format!(
            "{indent} {}{}|{} {indent_offset}{arrows}\n",
            color::Fg(color::Blue),
            style::Bold,
            color::Fg(color::Yellow),
        );

        res
    }
}

#[derive(Error, Debug, PartialEq, PartialOrd)]
pub enum ParseErrorKind {
    #[error("invalid parameter")]
    InvalidParameter,

    #[error("invalid file")]
    InvalidFile,

    #[error("invalid image")]
    InvalidImage,

    #[error("file not found")]
    NotFound,

    #[error("syntax")]
    Syntax,
}

pub struct SyntaxError {
    pub kind: SyntaxErrorKind,
    pub loc: FileLoc,
    pub msg: Option<String>,
}

impl SyntaxError {
    pub fn new(kind: SyntaxErrorKind, loc: FileLoc, msg: Option<String>) -> SyntaxError {
        SyntaxError { kind, loc, msg }
    }

    #[inline]
    pub fn with_loc(self, loc: FileLoc) -> SyntaxError {
        SyntaxError {
            kind: self.kind,
            loc,
            msg: self.msg,
        }
    }

    pub fn format(&self, s: &str) -> String {
        use termion::{color, style};

        let line_digits = self.loc.line.checked_ilog10().unwrap_or(0) + 1;
        let indent = " ".repeat(line_digits as usize);

        let mut res = format!(
            // error: [reason]
            //     --> [filename]:[line]:[col]
            //      |
            "{}{}error: {}{}\n{indent}{}-->{}{} {}:{}:{}\n{indent} {}{}|\n",
            color::Fg(color::Red),
            style::Bold,
            color::Fg(color::Reset),
            self.kind,
            color::Fg(color::Blue),
            color::Fg(color::Reset),
            style::Reset,
            self.loc.filename,
            self.loc.line,
            self.loc.start,
            color::Fg(color::Blue),
            style::Bold,
        );

        let line_contents = s.lines().nth(self.loc.line as usize).unwrap_or("");

        res += &format!(
            "{}{}{} |{}{} {}\n",
            color::Fg(color::Blue),
            style::Bold,
            self.loc.line,
            color::Fg(color::Reset),
            style::Reset,
            line_contents,
        );

        let indent_offset = " ".repeat(self.loc.start as usize);
        let arrows = "^".repeat(self.loc.len as usize);

        res += &format!(
            "{indent} {}{}|{} {indent_offset}{arrows}\n",
            color::Fg(color::Blue),
            style::Bold,
            color::Fg(color::Yellow),
        );

        res
    }
}

impl<E> From<E> for SyntaxError
where
    SyntaxErrorKind: From<E>
{
    fn from(value: E) -> Self {
        SyntaxError::new(SyntaxErrorKind::from(value), FileLoc::default(), None)
    }
}

#[derive(Error, Debug)]
pub enum SyntaxErrorKind {
    /// No more tokens.
    #[error("no tokens")]
    EndOfFile,

    /// Expected a token, but received `None`.
    #[error("token expected, got end of stream")]
    NoToken,

    #[error("failed to read file")]
    Io(#[from] io::Error),

    /// Token didn't pass basic validation checks.
    #[error("invalid token")]
    InvalidToken,

    /// Failed to parse string to float.
    #[error("unable to parse float")]
    ParseFloat(#[from] ParseFloatError),

    /// Failed to parse string to integer.
    #[error("unable to parse number")]
    ParseInt(#[from] ParseIntError),

    /// Failed to parse boolean.
    #[error("unable to parse bool")]
    ParseBool(#[from] ParseBoolError),

    /// Unable to cast from slice to array.
    #[error("unexpected number of arguments in array")]
    ParseSlice,

    /// Directive is unknown.
    #[error("unknown directive")]
    UnknownDirective(String),

    #[error("expected string token")]
    InvalidString,

    /// Failed to parse option's `[ value ]`
    #[error("unable to parse option value")]
    InvalidOptionValue,

    #[error("unknown coordinate system")]
    UnknownCoordinateSystem,

    #[error("invalid parameter name")]
    InvalidParamName,

    /// Unsupported parameter type.
    #[error("parameter type is invalid")]
    InvalidParamType,

    #[error("found duplicated parameter")]
    DuplicatedParamName,

    #[error("duplicated WorldBegin statement")]
    WorldAlreadyStarted,

    #[error("element is not allowed")]
    ElementNotAllowed,

    #[error("too many AttributeEnd")]
    TooManyEndAttributes,

    #[error("attempt to restore CoordSysTransform matrix with invalid name")]
    InvalidMatrixName,

    #[error("invalid camera type")]
    InvalidCameraType,

    #[error("unknown object type")]
    InvalidObjectType,

    #[error("unexpted token received")]
    UnexpectedToken,

    #[error("required param is missing")]
    MissingRequiredParameter,

    #[error("nested object attributes are not allowed")]
    NestedObjects,

    #[error("not found")]
    NotFound,
}
