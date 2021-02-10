# Contributing

Please follow these guidelines if you wish to contribute to Spektral.

## Bugs

If you found a bug in the latest version of Spektral, you can [open an issue](https://github.com/danielegrattarola/spektral/issues) to report it.

Before opening the issue, make sure to follow these steps:

1. Update to the current `develop` branch and see if the problem is already solved. Sometimes a change does not get released immediately on PyPi, so it might be a good idea to install from source.
2. Check old issues to see if the problem was already solved.
3. Make sure that your configuration matches all requirements, including:
    - Operating system
    - Python version
    - Tensorflow version (note that since version 0.3 Spektral only supports `tf.keras` and TensorFlow 2.1 and above)
4. Provide a minimal script to reproduce the issue. The script should be runnable as-is, with no modification or external data.
5. Include any stack trace/errors that you get.

If you want to try to fix the bug yourself, feel free to [open a pull request](https://github.com/danielegrattarola/spektral/pulls).

Bug fixes should be added to the `master` branch.

---

## Feature requests

If you want to request a feature, [open an issue](https://github.com/danielegrattarola/spektral/issues) or [a discussion](https://github.com/danielegrattarola/spektral/discussions) on GitHub and mark it as a feature request.

1. Give a detailed description of the feature, including why it is important and why it fits in the scope of the project.
Spektral is primarily a library for creating graph neural networks, so new features should gravitate around this subject.
2. Provide an example of the use case that you have in mind for your feature.

The quickest way to see your feature implemented is to code it yourself and then [open a pull request](https://github.com/danielegrattarola/spektral/pulls).

---

## Contributing a feature

New features should be added to the `develop` branch.
Contributions to the documentation go the `master` branch, instead.

There are no hard rules for contributing to Spektral, but you should try to follow these guidelines.

**General guidelines:**

- Use Black and iSort for formatting code (see the section below about pre-commit hooks);
- Make sure that the code you contribute is clearly identifiable in a PR (e.g., watch out for your IDE automatically reformatting files);
- New features should support:
    - Python >= 3.6
    - TensorFlow >= 2.1.0
    - Linux (at least Ubuntu >= 16.04) and MacOS >= 10.14
- Write tests for the new feature and then run:
    ```
    cd spektral
    pytest tests/
    ```
- Write docstrings for the new feature (copy the format of existing docstrings);

**Guidelines for adding new layers:**

- Message-passing/convolutional layers go in their own file in `layers/convolutional/`;
- Pooling layers go in their own file in `layers/pooling/`;
- Global pooling layers go in `layers/pooling/globalpool.py`;
- Layers should extend `MessagePassing`, `Conv` or `Pool`.
- Make sure that you understand [data modes](https://graphneural.network/data-modes/) and that you know the modes supported by your layer. Layers should support at least one of disjoint or batch mode;
- Write documentation for your layer, including any references to scientific literature, supported modes, mathematical formulation, input, outputs and arguments. 

**Guidelines for testing:**

- Tests are found in `tests/`;
- See the docstrings in each file for more information;

**Guidelines for the documentation:**

- See the documentation in the other classes/functions for how to format docstrings (it's important that the format is the same so that the docs can be built automatically);
- Docs are automatically generated using `docs/autogen.py`. Make sure to include any new layer as an entry in the `PAGES` dictionary. It is not necessary to add utils and other minor functions to `autogen.py` (although you should still write the docstrings).

**Pre-commit**

This package is configured for [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To install, use

```bash
pip install pre-commit
pre-commit install
```

This will install pre-commit hooks that will cause commits to fail if they do not pass certain linting/formatting constraints. It will automatically make formatting changes, though these changes will not be committed.
