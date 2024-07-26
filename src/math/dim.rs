#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    X,
    Y,
    Z,
    W,
}

impl std::fmt::Display for Dimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dimension::X => write!(f, "Dimension::X"),
            Dimension::Y => write!(f, "Dimension::Y"),
            Dimension::Z => write!(f, "Dimension::Z"),
            Dimension::W => write!(f, "Dimension::W"),
        }
    }
}
