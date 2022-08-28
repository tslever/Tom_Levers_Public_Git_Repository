//---------------------------------------------------------------------------
// BSTNode.java               by Dale/Joyce/Weems                   Chapter 7
//
// Implements nodes holding info of class <T> for a binary search tree.
//---------------------------------------------------------------------------

package support;

public class BSTNode<T>
{
  private T info;                // The node info
  private BSTNode<T> left;       // A link to the left child node
  private BSTNode<T> right;      // A link to the right child node

  public BSTNode(T info)
  {
    this.info = info; left = null;  right = null;
  }
 
  public void setInfo(T info){this.info = info;}
  public T getInfo(){return info;}
  
  public void setLeft(BSTNode<T> link){left = link;}
  public void setRight(BSTNode<T> link){right = link;}
  
  public BSTNode<T> getLeft(){return left;}
  public BSTNode<T> getRight(){return right;}
}
 
 