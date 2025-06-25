# ClassifierInference
## 如何运行
```
conan profile path default
cd ClassifierInference
cmake --preset conan-default
cmake -B ./build -DCMAKE_BUILD_TYPE=Release
cmake --build ./build/ --config Release
conan create .
```

