#pragma once
namespace src::code {
struct data {
  unsigned long key;
  double value = 0;
};
inline bool seed_compare(data a, data b) {
  return a.key < b.key;
}
// template <typename S> void merge_sort_in_thrust(S &s);
// template <typename S> void csp_in_thrust(S &s);
// template <typename S> void merge_sort(S &s);
// template <typename S> void csp(S &s);
void merge_sort_in_thrust(data* datas);
}