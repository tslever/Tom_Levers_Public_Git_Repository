box::register_S3_method('print', 'message', print.message)

print.message <- function(message) {
    cat(message)
}