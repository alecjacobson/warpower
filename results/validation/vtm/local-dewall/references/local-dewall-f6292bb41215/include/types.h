
enum Status {
    success = 0,
    cycles_overflow = 1,
    vertex_overflow = 2,
    need_exact_predict = 3,
    init_fail = 4


};


enum PredicateBounds {
    epsilon = 0,
    resulterrbound = 1,
    ccwerrboundA,
    ccwerrboundB,
    ccwerrboundC,
    isperrboundA,
    isperrboundB,
    isperrboundC,
    o3derrboundA,
    o3derrboundB,
    o3derrboundC,
    o3derrboundAlifted,
    numpredConsts
};

enum DPredicateSizes
{
    // Size of each array
    Temp96Size = 96,
    Det384xSize = 384,
    Det384ySize = 384,
    DetxySize = 768,
    AdetSize = 1152,
    BdetSize = 1152,
    AbdetSize = 2304,
    CdedetSize = 3456,

    // Total size
    PredicateTotalSize = 0
    +Temp96Size
    + Det384xSize
    + Det384ySize
    + DetxySize
    + AdetSize
    + BdetSize
    +AbdetSize
    + CdedetSize
};

enum Bittag {
    PROCESSED = 0,
    INEXACT = 1,
    OUTPUT = 2
};
#define BITPROCESSED 1
#define BITINEXACT 2
#define BITOUTPUT 4

static char StatusStr[5][30] = { "success", "cycles_overflow", "vertex_overflow", "need_exact_predict", "init_fail" };


static void show_status_stats(std::vector<uchar>& stat) {
    std::cout << std::endl << " ---------Summary of success/failure------------" << std::endl;
    std::vector<int> nb_statuss(5, 0);
    for (int i = 0; i < stat.size() ; i++) {
        nb_statuss[stat[i]]++;
         //if (stat[i] != success)
         //    std::cout<<i<< StatusStr[stat[i]] <<std::endl;
    }
    for (int r = 0; r < 5; r++)
        std::cout << " " << StatusStr[r] << "   " << nb_statuss[r] << std::endl;
    std::cout << " " << StatusStr[0] << "   " << nb_statuss[0] << " /  " << stat.size() << std::endl << std::endl;
}

