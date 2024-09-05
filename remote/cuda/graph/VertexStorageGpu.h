#pragma once

template<typename T>
struct VertexStorageGpu
{
    char* layout;
    unsigned* offsets;
    unsigned* indexesInH;
};

template<typename T>
struct EdgeStorageGpu
{
    T* edges;
};
