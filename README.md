# Project 2 Test Suite

To-do:

- [ ] Rewrite this to work with classes
- [ ] Explain how the interface works in the `README` and provide examples
- [ ] Create some skeleton code that students can use
- [ ] Maybe force students to read the `README` or like maybe a Wiki of sorts rather than giving an example
- [ ] Have the framework be able to just add an array to the data section

## Matrix Format

Matrices can be written by passing in a list of list of integers to the `Matrix` class.

```py
example_matrix: Matrix = Matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
```

In the case of the matrix above, the corresponding bytes in the `.bin` file would be

```bin
00000000: 00000003 00000003 00000001 00000002  ................
00000010: 00000003 00000004 00000005 00000006  ................
00000020: 00000007 00000008 00000009           ............
```

### `.to_bin`

Once you have a Matrix instance, you can then call `.to_bin` on it to get the corresponding `.bin` file associated with it.

### `.to_vector`

<!-- TODO -->

Can convert a matrix to a vector

## Vector

---