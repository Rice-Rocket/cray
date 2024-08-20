use super::token::Token;

pub(crate) struct Tokenizer<'a> {
    str: &'a str,
    offset: usize,
}

impl<'a> Tokenizer<'a> {
    pub fn new(str: &'a str) -> Self {
        Self { str, offset: 0 }
    }

    fn rewind_until(&mut self, chars: &[char]) -> usize {
        let mut offset = 0;

        loop {
            let Some(ch) = self.peek_char() else {
                break;
            };

            if chars.contains(&ch) {
                break;
            }

            if let Some((pos, _)) = self.next_char() {
                offset = pos;
            }
        }

        offset
    }

    fn peek_char(&mut self) -> Option<char> {
        if self.offset > self.str.len() {
            None
        } else {
            self.str[self.offset..].chars().next()
        }
    }

    fn next_char(&mut self) -> Option<(usize, char)> {
        match self.peek_char() {
            Some(ch) => {
                let offset = self.offset;
                self.offset += 1;
                Some((offset, ch))
            },
            None => None,
        }
    }

    pub fn peek_token(&mut self) -> Option<Token<'a>> {
        let offset = self.offset;
        let token = self.next();
        self.offset = offset;

        token
    }

    pub fn token(&self, start: usize, end: usize) -> Token<'a> {
        Token::new(&self.str[start..end])
    }

    pub fn offset(&self) -> usize {
        self.offset
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (start, ch) = self.next_char()?;

            let token = match ch {
                '[' | ']' => self.token(start, start + 1),
                ' ' | '\n' | '\t' | '\r' => continue,
                '"' => {
                    let mut end = self.rewind_until(&['"']);

                    if let Some((pos, _)) = self.next_char() {
                        end = pos;
                    }

                    self.token(start, end + 1)
                },
                '#' => {
                    self.rewind_until(&['\r', '\n']);
                    continue;
                },
                _ => {
                    let mut end = self.rewind_until(&[' ', '\r', '\n', '\t', '"', '[', ']']);
                    if end == 0 {
                        end = start;
                    }

                    self.token(start, end + 1)
                }
            };

            return Some(token);
        }
    }
}
