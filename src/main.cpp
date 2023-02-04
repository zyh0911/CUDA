
#include <src / utility / logger.hpp>
#include <src / utility / timer.hpp>
#include <vector>
#include <src/code/csp.hpp>
namespace src {
unsigned long const NUM_ELEMENT = (1 << 20);
struct data {
  unsigned long key;
  double value = 0;
};

struct initialize_array() {
  data datas[NUM_ELEMENT];
  for(unsigned long i = 0; i < NUM_ELEMENT; i++)  
    {
        datas[i].key = i;
    }  
    for(int i = 0; i < NUM_ELEMENT; i++)
    {
        swap(datas[rand()%7].key, datas[i].key);
    }
  return datas;
}

struct initialize_vector() {
  std::vector<data> datas;
  datas.reserve(NUM_ELEMENT);
  for(unsigned long i = 0; i < NUM_ELEMENT; i++)  
    {
        datas[i].key = i;
    }  
    for(int i = 0; i < NUM_ELEMENT; i++)
    {
        std::swap(datas[rand()%7].key, datas[i].key);
    }
  return datas;
}

int main() {
  src::util::logger logger("/dev/null", src::util::log_level::info, true);
  src::util::timer enc_check1("time1", logger);
  src::util::timer enc_check2("time2", logger);
  src::util::timer enc_check3("time3", logger);
  src::util::timer enc_check4("time4", logger);
}
}  // namespace src