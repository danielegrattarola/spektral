# Contributing

Please follow these guidelines if you wish to contribute to Spektral.
Improvements, new features, and bug fixes are more than welcome. 


## Bugs

If you found a bug in the latest version of Spektral, you can [open an issue](https://github.com/danielegrattarola/spektral/issues) to report it.

Before opening the issue, make sure to follow these steps: 

1. Update to the current master branch and see if the problem is already solved. Sometimes a change does not get released immediately on PyPi, so it might be a good idea to install from source. 
2. Check old issues to see if the problem was already solved. 
3. Make sure that your configuration checks all requirements. If you open an issue, provide as much detail as possible regarding your setup, including: 
    - Operating system
    - Python version
    - Version of Spektral
    - Version of Tensorflow and Keras (note that from version 0.2 spektral only supports `tf.keras`)
    - Version of CUDA and cuDNN
4. Provide a minimal script to reproduce the issue. The script should be runnable as-is, with no modification or external data. 
5. If the bug causes your programs to crash, provide an example of the output you get. If the output is too long, please use Pastebin or other equivalent services and share the link. 

If you want to try to fix the bug yourself, feel free to [open a pull request](https://github.com/danielegrattarola/spektral/pulls). Bug fixes should be added to the `master` branch.

---

## Features

If you want to request a feature, [open an issue](https://github.com/danielegrattarola/spektral/issues) on GitHub and clearly mark it as a feature request (e.g., use a tag like [Feature Request] in the title).

1. Give a detailed description of the feature, including what should the new functionality be, why it is important, and why it fits in the scope of the project. 
Spektral is primarily a library for creating graph neural networks, so new features should gravitate around this subject.
2. Provide a code/pseudocode example of the use case that you have in mind for your feature. 

If you feel up to the task, the quickest way to see your feature implemented is to code it yourself and then [open a pull request](https://github.com/danielegrattarola/spektral/pulls). New features should be added to the `develop` branch.

If you want to contribute to the documentation, follow the same process but open the PR on the `master` branch. 

See below for general contribution guidelines.

---

## Adding new examples

If you have coded an interesting example of how to use Spektral for a particular application, consider submitting it to the [examples page](spektral.graphneural.network/examples).

If you want to add an example, then you can open a PR on `master`, as described above.
You can also add a link to an external platform (like a Kaggle kernel).

---

## General guidelines

There are no hard rules for contributing to Spektral, but you should try to follow these guidelines: 

- Format your code according to PEP8;
- Make sure that your IDE does not reformat entire files automatically; make sure that the code you contribute is clearly identifiable in a PR.
- Make sure to configure your environment according to the requirements of the library. New features should run on the supported platforms, Python versions, and library versions. 
- Make sure that you write tests for what you contribute. Before opening a PR, run: 
    ```
    cd spektral
    pytest tests/
    ```

Here's a quick checklist for things to keep in mind: 

- Supported Python version: 3.5+
- Supported TensorFlow version: 
    - 2.0.0+ if adding a feature to Spektral releases after v0.2; 
    - 1.15.0 if fixing a bug on an earlier version
- Supported operating systems: Ubuntu 16.04+, MacOS 10.14+


