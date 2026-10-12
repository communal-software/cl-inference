# cl-inference

> **Work in progress.** This project is under development; expect missing features and breaking changes.

One `complete` contract for hosted and local LLMs in Common Lisp. `cl-inference/local` runs models in-process on [cl-tensor](https://github.com/communal-software/cl-tensor); its architectures are declared in a Lisp DSL where a similar architecture inherits from an existing one and overrides what differs.

## Docs

- [Design](docs/design.md)
- [Architectures](docs/architectures.md)
- [Blocks](docs/blocks.md)
- [Weights](docs/weights.md)
- [GGUF](docs/gguf.md)
- [Client](docs/client.md)
- [Testing](docs/testing.md)
- [Limitations](docs/limitations.md)

Work is tracked at <https://github.com/communal-software/cl-inference/issues>.

## License

[GPLv3](LICENSE).
