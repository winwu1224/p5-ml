#include "BinarySearchTree.hpp"
#include "unit_test_framework.hpp"
#include <sstream>

using namespace std;

TEST(test_null) {
    BinarySearchTree<int> tree;
    ASSERT_TRUE(tree.empty());
    ASSERT_TRUE(tree.size() == 0);
    ASSERT_TRUE(tree.height() == 0);
}

TEST(test_funcs) {
  BinarySearchTree<int> tree;
  tree.insert(6);
  ASSERT_FALSE(tree.empty());
  ASSERT_TRUE(tree.size() == 1);
  ASSERT_TRUE(tree.height() == 1);

  ASSERT_TRUE(tree.find(6) != tree.end());

  tree.insert(8);
  tree.insert(4);

  ASSERT_TRUE(tree.check_sorting_invariant());
  ASSERT_TRUE(*tree.max_element() == 8);
  ASSERT_TRUE(*tree.min_element() == 4);
  ASSERT_TRUE(*tree.min_greater_than(6) == 8);

  cout << tree.to_string() << endl << endl;
  cout << tree << endl << endl;

  ostringstream oss_preorder;
  tree.traverse_preorder(oss_preorder);
  cout << oss_preorder.str() << endl << endl;
  ASSERT_TRUE(oss_preorder.str() == "6 4 8 ");

  ostringstream oss_inorder;
  tree.traverse_inorder(oss_inorder);
  cout << oss_inorder.str() << endl << endl;
  ASSERT_TRUE(oss_inorder.str() == "4 6 8 ");
}

TEST_MAIN()
