package Com.TSL.ShoppingListUtilities;


/**
* @author Yingjin Cui
* version 1.0
* since   2020-03
*
* Item.java: Item class represents grocery items to be added to the shopping list
*/

public class Item implements Comparable<Item>{
   private String name; //product name
   private int quantity; //quantity
   private double unitPrice; //unit price

   public Item(String name, int quantity, double unitPrice){
      setName(name); 
      setQuantity(quantity); 
      setUnitPrice(unitPrice); 
   }

   public String getName(){
      return name;
   }
   public int getQuantity(){
      return quantity;
   }
   public double getUnitPrice(){
      return unitPrice;
   }
   public void setName(String name){
      // The length of product name is limited to [1,50], if more than 50 characters, use the first 50 chars 
      // must neigher be a null value nor an empty String (throw runtime exception)
      if(name ==null || name.length()==0){
         throw new RuntimeException("null or empty String value is not allowed");
      }
      if(name.length()>50){
         name=name.substring(0, 50);
      }
      this.name = name;
   }
   public void setQuantity(int quantity){
      //quantity is limited to [1, 100], assign 1 for all invalid numbers
      if(quantity<1 || quantity>100){
         quantity=1;
      }
      this.quantity = quantity;
   }
   public void setUnitPrice(double unitPrice){
      //unitPrice is limited to (0, 1000], assign 0.99 for all invalid numbers
      if(unitPrice<=0 || unitPrice>1000){
         unitPrice=0.99;
      }
      this.unitPrice = unitPrice;
   }
   public int compareTo(Item item){
      //compare two items based on their names, ignore cases
      return name.toLowerCase().compareTo(item.getName().toLowerCase());
   }
   public double subtotal(){
      return quantity*unitPrice;
   }
   public String toString(){
      return String.format("%-50s%-5d$%.2f",name,quantity, quantity*unitPrice);
   }
}