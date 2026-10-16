#include "llama.h"
#include "unicode.h"

#include <iostream>
#include <iterator>
#include <string>
#include <sstream>
#include <vector>

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--split") {
        const std::string text(std::istreambuf_iterator<char>(std::cin), {});
        const auto pieces = unicode_regex_split(text, {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, false);
        constexpr char hex[] = "0123456789abcdef";
        for (const auto & piece : pieces) {
            for (unsigned char byte : piece) std::cout << hex[byte >> 4] << hex[byte & 15];
            std::cout << '\n';
        }
        return 0;
    }
    if (argc != 2) return 2;
    llama_backend_init();
    auto params = llama_model_default_params();
    params.vocab_only = true;
    auto * model = llama_model_load_from_file(argv[1], params);
    if (!model) return 3;
    const auto * vocab = llama_model_get_vocab(model);
    std::string request;
    while (std::getline(std::cin, request)) {
        std::istringstream fields(request);
        bool add, special;
        std::string encoded, text;
        if (!(fields >> add >> special)) return 4;
        fields >> encoded;
        if (encoded.size() % 2) return 5;
        for (size_t i = 0; i < encoded.size(); i += 2) {
            text.push_back(static_cast<char>(std::stoul(encoded.substr(i, 2), nullptr, 16)));
        }
        int count = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, special);
        std::vector<llama_token> ids(count < 0 ? -count : count);
        count = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), false, special);
        if (count < 0) return 6;
        // llama_tokenize's add_special follows metadata; Lisp's add-bos explicitly forces BOS.
        if (add) {
            const auto bos = llama_vocab_bos(vocab);
            if (bos == LLAMA_TOKEN_NULL) return 8;
            ids.insert(ids.begin(), bos);
            ++count;
        }
        for (int i = 0; i < count; ++i) std::cout << ids[i] << ' ';
        std::cout << '\n';
        constexpr char hex[] = "0123456789abcdef";
        for (int i = 0; i < count; ++i) {
            int size = llama_token_to_piece(vocab, ids[i], nullptr, 0, 0, special);
            std::vector<char> piece(size < 0 ? -size : size);
            size = llama_token_to_piece(vocab, ids[i], piece.data(), piece.size(), 0, special);
            if (size < 0) return 7;
            for (int j = 0; j < size; ++j) {
                auto byte = static_cast<unsigned char>(piece[j]);
                std::cout << hex[byte >> 4] << hex[byte & 15];
            }
        }
        std::cout << '\n';
    }
    llama_model_free(model);
    llama_backend_free();
}
