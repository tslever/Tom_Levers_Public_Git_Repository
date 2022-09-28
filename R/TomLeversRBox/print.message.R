#box::export() # Use to export nothing and disable attaching (https://klmr.me/box/reference/export.html)
print.message <- function(message) {
    cat(message)
}

box::register_S3_method('print', 'message', print.message)