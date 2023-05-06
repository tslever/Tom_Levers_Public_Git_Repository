
// Allows getSearchParameters to find class SearchParameters.
package com.mycompany.serverutilities.productutilities;

/**
 * Defines class SearchParameters, which will be extracted from an ice cream
 * application message and will inform getting the appropriate ice cream
 * products.
 * @version 0.0
 * @author Tom Lever
 */
public class SearchParameters {
    
    private final String[] ingredientsList;
    private final int lengthOfIngredientsList;
    
    /**
     * Defines constructor SearchParameters, which will set this.ingredientsList
     * as ingredientsListToUse.
     * @param ingredientsListToUse
     * @param lengthOfIngredientsListToUse
     */
    public SearchParameters(
        String[] ingredientsListToUse, int lengthOfIngredientsListToUse) {
        this.ingredientsList = ingredientsListToUse;
        this.lengthOfIngredientsList = lengthOfIngredientsListToUse;
    }
    
    
    /**
     * Defines getIngredientsList, which returns this.ingredientsList.
     * @return this.ingredientsList.
     */
    public String[] getIngredientsList() {
        return this.ingredientsList;
    }
    
    /**
     * Defines getLengthOfIngredientsList, which returns
     * this.lengthOfIngredientsList.
     * @return this.lengthOfIngredientsList
     */
    public int getLengthOfIngredientsList() {
        return this.lengthOfIngredientsList;
    }
}
