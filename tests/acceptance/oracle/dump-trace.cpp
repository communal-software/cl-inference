// Dev-only llama.cpp oracle; never linked into the Lisp runtime.
#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

struct dump_state {
    std::ofstream trace;
    bool          active = true;
};

static bool wanted(const char * name) {
    return std::string(name) == "result_output" || std::string(name).rfind("l_out-", 0) == 0;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = (dump_state *) user_data;
    if (!wanted(t->name)) {
        return ask ? false : true;
    }
    if (ask) {
        return state->active;
    }
    if (!state->active || t->type != GGML_TYPE_F32) {
        return true;
    }
    std::vector<uint8_t> copy;
    const uint8_t * data = (const uint8_t *) t->data;
    if (!ggml_backend_buffer_is_host(t->buffer)) {
        copy.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, copy.data(), 0, copy.size());
        data = copy.data();
    }
    const int64_t row = t->ne[1] - 1;
    state->trace << t->name << ' ' << t->ne[0];
    state->trace.precision(9);
    for (int64_t i = 0; i < t->ne[0]; i++) {
        state->trace << ' ' << *(const float *) (data + row * t->nb[1] + i * t->nb[0]);
    }
    state->trace << '\n';
    return true;
}

int main(int argc, char ** argv) {
    const char * dir_env = std::getenv("DUMP_DIR");
    if (!dir_env) {
        fprintf(stderr, "DUMP_DIR is required\n");
        return 1;
    }
    const std::string dir = dir_env;
    const int n_greedy = std::getenv("DUMP_N") ? std::atoi(std::getenv("DUMP_N")) : 8;

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();

    dump_state state;
    state.trace.open(dir + "/trace.txt");
    params.cb_eval           = cb_eval;
    params.cb_eval_user_data = &state;
    params.warmup            = false;

    auto init  = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx   = init->context();
    if (!model || !ctx) {
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::ofstream decode;
    if (std::getenv("DUMP_DECODE_LOGITS")) decode.open(dir + "/decode.f32", std::ios::binary);
    auto save_logits = [&]() {
        if (decode.is_open()) {
            decode.write(reinterpret_cast<const char *>(llama_get_logits_ith(ctx, -1)), n_vocab * sizeof(float));
        }
    };

    std::vector<llama_token> tokens;
    const char * messages_file = std::getenv("DUMP_MESSAGES");
    const char * user = std::getenv("DUMP_USER");
    if (messages_file || user) {
        common_chat_templates_inputs inputs;
        if (messages_file) {
            std::ifstream stream(messages_file);
            for (const auto & item : nlohmann::ordered_json::parse(stream)) {
                common_chat_msg message;
                message.role = item.at("role").get<std::string>();
                message.content = item.at("content").get<std::string>();
                inputs.messages.push_back(message);
            }
        } else {
            if (const char * system = std::getenv("DUMP_SYSTEM")) {
                common_chat_msg message;
                message.role = "system";
                message.content = system;
                inputs.messages.push_back(message);
            }
            common_chat_msg message;
            message.role = "user";
            message.content = user;
            inputs.messages.push_back(message);
        }
        inputs.add_generation_prompt = true;
        if (const char * thinking = std::getenv("DUMP_THINKING")) {
            inputs.enable_thinking = std::string(thinking) != "0";
        }
        inputs.chat_template_kwargs["date_string"] = "\"26 Jul 2024\"";
        auto templates = common_chat_templates_init(model, "");
        const std::string prompt = common_chat_templates_apply(templates.get(), inputs).prompt;
        std::ofstream(dir + "/prompt.txt") << prompt;
        tokens = common_tokenize(ctx, prompt, llama_vocab_get_add_bos(vocab), true);
    } else {
        tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    }
    {
        std::ofstream ids(dir + "/prompt.ids");
        for (auto id : tokens) ids << id << '\n';
    }

    auto argmax = [&](const float * logits) {
        return (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
    };

    {
        common_batch batch = common_batch_get_one(ctx, tokens);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            return 1;
        }
    }
    state.active = false;
    save_logits();
    const float * logits = llama_get_logits_ith(ctx, -1);
    {
        std::vector<int> order(n_vocab);
        for (int i = 0; i < n_vocab; i++) order[i] = i;
        std::partial_sort(order.begin(), order.begin() + 10, order.end(),
                          [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
        std::ofstream top(dir + "/top.txt");
        top.precision(9);
        for (int i = 0; i < 10; i++) top << order[i] << ' ' << logits[order[i]] << '\n';
    }

    std::ofstream greedy(dir + "/greedy.ids");
    llama_token next = argmax(logits);
    for (int i = 0; i < n_greedy; i++) {
        greedy << next << '\n';
        if ((i + 1 == n_greedy && !decode.is_open()) || llama_vocab_is_eog(vocab, next)) break;
        common_batch batch = common_batch_get_one(ctx, std::vector<llama_token>{next});
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            return 1;
        }
        save_logits();
        next = argmax(llama_get_logits_ith(ctx, -1));
    }
    llama_backend_free();
    return 0;
}
