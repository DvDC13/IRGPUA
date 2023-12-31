#pragma once

#include <iostream>

struct ParsingArgs
{
    enum Version
    {
        CPU,
        GPU,
        THRUST
    };

    Version version;
    bool benchmark;
};

ParsingArgs parse_args(int argc, char** argv);