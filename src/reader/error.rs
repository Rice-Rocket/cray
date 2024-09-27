use std::{io, num::{ParseFloatError, ParseIntError}, str::ParseBoolError};

use thiserror::Error;

use crate::Wrap;

use super::target::FileLoc;

pub type ParseResult<T, const W: bool> = std::result::Result<ParseOk<T, W>, ParseError>;

pub struct ParseDiagnostic {
    pub msg: Option<String>,
    pub loc: FileLoc,
}

pub struct ParseOk<T, const W: bool> {
    pub v: T,
    pub warns: Vec<ParseDiagnostic>,
}

impl<T, const W: bool> ParseOk<T, W> {
    #[inline]
    pub const fn new(v: T) -> ParseOk<T, W> {
        ParseOk { v, warns: vec![] }
    }

    #[inline]
    pub const fn ok(v: T) -> ParseResult<T, W> {
        Ok(Self::new(v))
    }
}

impl<T> ParseOk<T, false> {
    #[inline]
    pub fn unwrap(self) -> T {
        self.v
    }
}

impl<T> ParseOk<T, true> {
    #[inline]
    pub fn split(self) -> (T, ParseOk<(), true>) {
        (self.v, ParseOk::<(), true> { v: (), warns: self.warns })
    }
}

pub struct ParseError {
    pub errs: Vec<(ParseErrorKind, ParseDiagnostic)>,
}

impl ParseError {
    #[inline]
    pub fn new(kind: ParseErrorKind, loc: FileLoc, msg: Option<String>) -> ParseError {
        ParseError { errs: vec![(kind, ParseDiagnostic { msg, loc })] }
    }

    #[inline]
    pub fn last_loc(&self) -> Option<&FileLoc> {
        self.errs.last().map(|e| &e.1.loc)
    }

    #[inline]
    pub fn last_loc_mut(&mut self) -> Option<&mut FileLoc> {
        self.errs.last_mut().map(|e| &mut e.1.loc)
    }
}

pub enum ParseErrorKind {

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
