
#include <src/utility/logger.hpp>
#include <src/utility/timer.hpp>
#include <src/code/csp.hpp>
#include <vector>
namespace src {
unsigned long const NUM_ELEMENT = (1 << 20);

template<class T> void swap(T &x, T &y){ T tmp1 = x; x = y; y = tmp1; }

code::data* initialize_array() {
  code::data* d = new code::data[NUM_ELEMENT];
  for (unsigned long i = 0; i < NUM_ELEMENT; i++) {
    d[i].key = i;
  }
  for (int i = 0; i < NUM_ELEMENT; i++) {
    swap(d[rand() % 7].key, d[i].key);
  }
  return d;
}

// struct initialize_vector() {
//   std::vector<data> datas;
//   datas.reserve(NUM_ELEMENT);
//   for(unsigned long i = 0; i < NUM_ELEMENT; i++)
//     {
//         datas[i].key = i;
//     }
//     for(int i = 0; i < NUM_ELEMENT; i++)
//     {
//         std::swap(datas[rand()%7].key, datas[i].key);
//     }
//   return datas;
// }

int main() {
  src::util::logger logger("/dev/null", src::util::log_level::info, true);
  src::util::timer  enc_check1("time1", logger);
  src::util::timer  enc_check2("time2", logger);
  src::util::timer  enc_check3("time3", logger);
  src::util::timer  enc_check4("time4", logger);
  auto              d = initialize_array();
  code::merge_sort_in_thrust(d);
  return 0;
}
}  // namespace src