package Com.TSL.UtilitiesForCalculatingVocabularyDensityAndWordFrequencies;


/**
* @author YINGJIN CUI
* @version 1.0
* since   2020-03
*
* LLNode.java: LLNode class 
*/

public class LLNode<T>{

   private T data;
   private LLNode<T> next;

   
   /**
    * LLNode(T data) represents a one-parameter constructor for LLNode, which sets the data of this linked-list node to
    * the provided data, and sets this linked-list node's reference to null.
    * 
    * @param data
    */
   
   public LLNode(T data){
      setData(data);
      next=null;
   }
   
   
   /**
    * LLNode(T data, LLNode<T> next) represents a two-parameter constructor for LLNode, which sets the data of this
    * linked-list node to the provided data, and sets this linked-list node's reference to the provided reference.
    * @param data
    * @param next
    */
   
   public LLNode(T data, LLNode<T> next){
      setData(data); 
      setNext(next); 
   }

   
   /**
    * getData provides this linked-list node's data.
    * @return
    */
   
   public T getData(){
      return data;
   }
   
   
   /**
    * getNext provides this linked-list node's reference.
    * @return
    */
   
   public LLNode<T> getNext(){
      return next;
   }
   
   
   /**
    * setData sets this linked-list node's data with the provided data.
    * @param data
    */
   
   public void setData(T data){
      this.data = data;
   }
   
   
   /**
    * setNext sets this linked-list node's reference to the provided reference.
    * @param next
    */
   
   public void setNext(LLNode<T> next){
      this.next = next;
   }

}