#include "common.h"
#include "ggml-cpp.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"

// Test demonstrating KV cache shifting issue:
// - Works with gemma3 + images
// - Fails with qwen2vl + images

namespace {
    std::string get_model_arch(const std::string& path) {
        auto gguf_ctx = gguf_context_ptr(
            gguf_init_from_file(path.c_str(), {.no_alloc = false, .ctx = nullptr}));

        int64_t key_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
        if (key_id < 0) {
            throw std::runtime_error("Architecture key not found");
        }
        return gguf_get_val_str(gguf_ctx.get(), key_id);
    }

    std::string build_prompt(const std::string& arch) {
        if (arch == "qwen3vl") {
            return "<|im_start|>user\n<__media__><__media__>what is in the image?<|im_end|>\n<|im_start|>assistant\n";
        } else if (arch == "gemma3") {
            return "<start_of_turn>user\n<__media__><__media__><end_of_turn>\n<start_of_turn>model\n";
        }
        throw std::runtime_error("Unsupported architecture: " + arch);
    }

    void generate_tokens(llama_context* ctx, llama_model* model, common_sampler* sampler,
                        llama_pos& n_pos, int count, const std::string& phase) {
        printf("\n=== Generation: %s ===\n", phase.c_str());

        llama_batch batch = llama_batch_init(1, 0, 1);

        for (int i = 0; i < count; i++) {
            llama_token token = common_sampler_sample(sampler, ctx, -1);
            common_sampler_accept(sampler, token, true);

            std::string token_str = common_token_to_piece(ctx, token);
            printf("%s", token_str.c_str());
            fflush(stdout);

            if (llama_vocab_is_eog(llama_model_get_vocab(model), token)) {
                printf(" [EOS]");
                break;
            }

            common_batch_clear(batch);
            common_batch_add(batch, token, n_pos++, {0}, true);

            if (llama_decode(ctx, batch)) {
                printf(" [DECODE_FAILED]");
                break;
            }
        }

        llama_batch_free(batch);
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <model_path> <mmproj_path>\n", argv[0]);
        return 1;
    }

    common_init();

    // Get model architecture
    std::string arch = get_model_arch(argv[1]);
    printf("Model architecture: %s\n", arch.c_str());

    if (arch != "qwen3vl" && arch != "gemma3") {
        fprintf(stderr, "Only qwen3vl and gemma3 are supported\n");
        return 1;
    }

    // Initialize model
    common_params params;
    params.model.path = argv[1];
    params.n_ctx = 1024;
    params.sampling.temp = 0;
    auto [model_ptr, ctx_ptr, _] = common_init_from_params(params);

    // Initialize vision projector
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = true;
    mparams.n_threads = 6;
    auto mtmd_ctx = mtmd::context_ptr(mtmd_init_from_file(argv[2], model_ptr.get(), mparams));

    // Load two copies of the same image
    mtmd::bitmap img1(mtmd_helper_bitmap_init_from_file(mtmd_ctx.get(), DICE_IMAGE_PATH));
    mtmd::bitmap img2(mtmd_helper_bitmap_init_from_file(mtmd_ctx.get(), DICE_IMAGE_PATH));
    if (!img1.ptr || !img2.ptr) {
        printf("Failed to load image\n");
        return 1;
    }

    // Tokenize prompt with both images
    std::string prompt = build_prompt(arch);
    mtmd_input_text text = {prompt.c_str(), true, true};
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    std::vector<const mtmd_bitmap*> images = {img1.ptr.get(), img2.ptr.get()};

    if (mtmd_tokenize(mtmd_ctx.get(), chunks.ptr.get(), &text, images.data(), images.size())) {
        printf("Tokenization failed\n");
        return 1;
    }

    // Evaluate prompt with both images
    llama_pos n_pos = 0;
    printf("Evaluating prompt with 2 images...\n");
    if (mtmd_helper_eval_chunks(mtmd_ctx.get(), ctx_ptr.get(), chunks.ptr.get(),
                               n_pos, 0, params.n_batch, true, &n_pos)) {
        printf("Evaluation failed\n");
        return 1;
    }
    printf("Position after mtmd evaluation: %d\n", n_pos);

    // Generate some tokens before shifting
    common_sampler* sampler = common_sampler_init(model_ptr.get(), params.sampling);
    generate_tokens(ctx_ptr.get(), model_ptr.get(), sampler, n_pos, 5, "Before cache shift");

    // Remove first image from KV cache
    constexpr llama_pos PREFIX_LEN = 3;  // Text tokens before first image
    // image n_pos calculated as such:
    //  - 1 pos for "<|vision_start|>" or "<start_of_image>"
    //  - 1 pos for qwen2vl image itself (MRoPE), 256 pos for gemma3
    //  - 1 pos for "<|vision_end|>" or "<end_of_image>"
    const llama_pos first_image_npos = arch == "qwen3vl" ? 3 : 258;
    constexpr llama_pos REMOVE_START = PREFIX_LEN;
    const llama_pos remove_end = PREFIX_LEN + first_image_npos;

    printf("\n=== Shift ===\n");
    llama_memory_t mem = llama_get_memory(ctx_ptr.get());
    printf("Next available cache pos before shift: %d\n", llama_memory_seq_pos_max(mem, 0) + 1);
    printf("Removing first image: positions %d-%d (%d tokens)\n",
           REMOVE_START, remove_end-1, first_image_npos);

    llama_memory_seq_rm(mem, 0, REMOVE_START, remove_end);

    printf("Shifting remaining tokens back by %d positions\n", first_image_npos);
    llama_memory_seq_add(mem, 0, remove_end, -1, -first_image_npos);

    n_pos = llama_memory_seq_pos_max(mem, 0) + 1;
    printf("New n_pos (next available cache pos) after shift: %d\n", n_pos);

    // Generate tokens after shifting - this should work for gemma3 but fail for qwen2vl
    generate_tokens(ctx_ptr.get(), model_ptr.get(), sampler, n_pos, 50, "After cache shift");

    common_sampler_free(sampler);
    return 0;
}