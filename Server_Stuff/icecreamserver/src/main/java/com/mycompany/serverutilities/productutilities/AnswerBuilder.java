
// Allows handle to find class AnswerBuilder.
package com.mycompany.serverutilities.productutilities;

// Imports classes.
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Defines class AnswerBuilder, static methods of which will be used to put
 * info or products into an Answer.
 * @version 0.0
 * @author Tom Lever
 */
public class AnswerBuilder {
   
    /**
     * Defines static method buildAnswerWithInfo, which returns an answer based
     * on infoToUse.
     * @param infoToUse
     * @return 
     */
    public static Answer buildAnswerWithInfo(String infoToUse) {
        Answer answer = new Answer();
        answer.putInHashMapAt("info", infoToUse);
        return answer;
    }
    
    /**
     * Defines static method buildAnswerWithProducts, which returns an answer
     * based on productsToUse.
     * @param arrayListOfProductsToUse
     * @return 
     */
    public static Answer buildAnswerWithProducts(
        ArrayList<Product> arrayListOfProductsToUse) {
        
        String productsInJSONFormat = "[";
        for (int i = 0; i < arrayListOfProductsToUse.size()-1; i++) {
            productsInJSONFormat +=
                arrayListOfProductsToUse.get(i).toString() + ", ";
        }
        productsInJSONFormat +=
            arrayListOfProductsToUse
            .get(arrayListOfProductsToUse.size()-1)
            .toString();
        productsInJSONFormat += "]";
        
        Answer answer = new Answer();
        answer.putInHashMapAt(
            "products", productsInJSONFormat);
        return answer;
    }
    
}
