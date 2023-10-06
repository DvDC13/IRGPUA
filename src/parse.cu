#include "parse.hh"

ParsingArgs parse_args(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " [--version <cpu|gpu>] [--benchmark]" << std::endl;
        exit(EXIT_FAILURE);
    }

    ParsingArgs args;
    args.benchmark = false;

    // Iterate over all arguments
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        // Check if the argument is a flag
        if (arg[0] == '-')
        {
            // Check if the flag is --version
            if (arg == "--version")
            {
                // Check if the version is cpu or gpu
                if (i + 1 < argc && std::string(argv[i + 1]) == "cpu")
                {
                    args.version = ParsingArgs::Version::CPU;
                    i++;
                }
                else if (i + 1 < argc && std::string(argv[i + 1]) == "gpu")
                {
                    args.version = ParsingArgs::Version::GPU;
                    i++;
                }
                else
                {
                    std::cerr << "Error: --version must be followed by cpu or gpu" << std::endl;
                    std::cerr << "Usage: " << argv[0] << " [--version <cpu|gpu>] [--benchmark]" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
            // Check if the flag is --benchmark
            else if (arg == "--benchmark")
            {
                args.benchmark = true;
            }
            // Unknown flag
            else
            {
                std::cerr << "Error: unknown flag " << arg << std::endl;
                std::cerr << "Usage: " << argv[0] << " [--version <cpu|gpu>] [--benchmark]" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        // Unknown argument
        else
        {
            std::cerr << "Error: unknown argument " << arg << std::endl;
            std::cerr << "Usage: " << argv[0] << " [--version <cpu|gpu>] [--benchmark]" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return args;
}