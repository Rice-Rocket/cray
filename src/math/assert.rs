#[cfg(feature = "math-assert")]
#[macro_export]
macro_rules! math_assert {
    ($($arg:tt)*) => ( assert!($($arg)*); )
}

#[cfg(not(feature = "math-assert"))]
#[macro_export]
macro_rules! math_assert {
    ($($arg:tt)*) => {};
}
