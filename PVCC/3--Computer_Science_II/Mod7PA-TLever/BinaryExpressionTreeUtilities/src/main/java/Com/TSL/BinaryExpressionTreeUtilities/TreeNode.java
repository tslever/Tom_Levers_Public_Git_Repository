package Com.TSL.BinaryExpressionTreeUtilities;


/**
* @author YINGJIN CUI
* @version 1.0
* since   2020-05
*
* TreeNode.java: TreeNode class.
*/

public class TreeNode<T>
   {
	
      private T value; 
      private TreeNode<T> left, right;
   
      
      /**
       * TreeNode(T initValue) is a one-parameter constructor for TreeNode, which sets this node's value to a provided
       * value, sets this Node's reference to a left child node to null, and sets this Node's reference to a right child
       * node to null.
       * 
       * @param data
       */
      
      public TreeNode(T initValue)
      { 
         value = initValue; 
         left = null; 
         right = null; 
      }
   
      
      /**
       * TreeNode(T initValue, TreeNode<T> initLeft, TreeNode<T> initRight) is a three-parameter constructor for
       * TreeNode, which sets this node's value to a provided value, sets this node's reference to a left child node to
       * a provided reference, and sets this node's reference to a right child node to a provided reference.
       * 
       * @param initValue
       * @param initLeft
       * @param initRight
       */
      
      public TreeNode(T initValue, TreeNode<T> initLeft, TreeNode<T> initRight)
      { 
         value = initValue; 
         left = initLeft; 
         right = initRight; 
      }
      
   
      /**
       * getValue provides this node's value.
       * 
       * @return
       */
      
      public T getValue()
      { 
         return value; 
      }
   
      
      /**
       * getLeft provides this node's reference to a left child node.
       * @return
       */
      
      public TreeNode<T> getLeft() 
      { 
         return left; 
      }
      
      
      /**
       * getRight provides this node's reference to a right child node.
       * @return
       */
   
      public TreeNode<T> getRight() 
      { 
         return right; 
      }
   
      
      /**
       * setValue sets this node's value.
       * @param theNewValue
       */
      
      public void setValue(T theNewValue) 
      { 
         value = theNewValue; 
      }
      
      
      /**
       * setLeft sets the reference of this node's left child to point to another node.   
       * @param theNewLeft
       */
      
      public void setLeft(TreeNode<T> theNewLeft) 
      { 
         left = theNewLeft;
      }
   
      
      /**
       * setRight sets the reference of this node's right child to point to another node.
       * @param theNewRight
       */
      
      public void setRight(TreeNode<T> theNewRight)
      { 
         right = theNewRight;
      }
   }