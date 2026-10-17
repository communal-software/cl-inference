#include "llama.h"

#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    llama_backend_init();
    auto params = llama_model_default_params();
    params.vocab_only = true;
    auto * model = llama_model_load_from_file(argv[1], params);
    if (!model) return 3;
    const auto * vocab = llama_model_get_vocab(model);
    const std::string text(std::istreambuf_iterator<char>(std::cin), {});
    const bool add = std::string(argv[2]) == "1";
    const bool special = std::string(argv[3]) == "1";
    int count = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, add, special);
    std::vector<llama_token> ids(count < 0 ? -count : count);
    count = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), add, special);
    if (count < 0) return 4;
    for (int i = 0; i < count; ++i) std::cout << ids[i] << ' ';
    std::cout << '\n';
    constexpr char hex[] = "0123456789abcdef";
    for (int i = 0; i < count; ++i) {
        int size = llama_token_to_piece(vocab, ids[i], nullptr, 0, 0, special);
        std::vector<char> piece(size < 0 ? -size : size);
        size = llama_token_to_piece(vocab, ids[i], piece.data(), piece.size(), 0, special);
        if (size < 0) return 5;
        for (int j = 0; j < size; ++j) {
            auto byte = static_cast<unsigned char>(piece[j]);
            std::cout << hex[byte >> 4] << hex[byte & 15];
        }
    }
    std::cout << '\n';
    llama_model_free(model);
    llama_backend_free();
}
