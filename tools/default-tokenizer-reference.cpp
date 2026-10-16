#include "unicode.h"

#include <iostream>
#include <iterator>
#include <string>

int main() {
    const std::string text(std::istreambuf_iterator<char>(std::cin), {});
    const auto pieces = unicode_regex_split(text, {
        "[\\p{P}\\$\\+<=>\\^~\\|]+",
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        "\\p{N}+",
        "[0-9][0-9][0-9]",
    }, false);
    constexpr char hex[] = "0123456789abcdef";
    for (const auto & piece : pieces) {
        for (unsigned char byte : piece) {
            std::cout << hex[byte >> 4] << hex[byte & 15];
        }
        std::cout << '\n';
    }
}
