package Com.TSL.UtilitiesForQueuingAndEvaluatingRandomNumbers;


/** *************************************
* Implements <T> nodes for a Linked List.
* 
* @author Emilia Butu
* @version 1.0
* @since 06/09/21
************************************** */

public class LLNode<T>
{
	
  protected LLNode<T> link;
  protected T info;

  
  /** -------------------------------------------------------------------------------------------------------------
   * LLNode is the one-parameter constructor for LLNode, which sets the info of this LLNode to the argument and the
   * link of this LLNode to null.
   * 
   * @param info
   ------------------------------------------------------------------------------------------------------------- */
  
  public LLNode(T info)
  {
    this.info = info;
    link = null;
  }

  
  /** -----------------------------------------------------
   * setInfo sets the info of this LLNode to provided info.
   * @param info
   ----------------------------------------------------- */
  
  public void setInfo(T info) {
	  this.info = info;
  }
  
  
  /** -----------------------------------
   * getInfo provides this LLNode's info.
   * @return
   ----------------------------------- */
  
  public T getInfo() {
	  return info;
  }
  
  
  /** -----------------------------------------------------------------------------------------
   * setLink sets the link of this LLNode to reference the provided LLNode with info of type T.
   * @param link
   ----------------------------------------------------------------------------------------- */
  
  public void setLink(LLNode<T> link) {
	  this.link = link;
  }
  
  
  /** -----------------------------------
   * getLink provides this LLNode's link.
   * @return
   ----------------------------------- */
  
  public LLNode<T> getLink() {
	  return link;
  }

}