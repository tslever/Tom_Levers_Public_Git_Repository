
// Allows getSearchCriteria to find class SearchCriteria.
package com.mycompany.serverutilities.productutilities;

/**
 * Defines class SearchCriteria, which will be extracted from an ice cream
 * application message and will inform getting the appropriate ice cream
 * products.
 * @version 0.0
 * @author Tom Lever
 */
public class SearchCriteria {
    
    private final String[] ingredientsList;
    
    /**
     * Defines constructor SearchCriteria, which will set this.ingredientsList
     * as ingredientsListToUse.
     * @param ingredientsListToUse
     */
    public SearchCriteria(String[] ingredientsListToUse) {
        this.ingredientsList = ingredientsListToUse;
    }
    
    
    /**
     * Defines getIngredientsList, which will return this.ingredientsList.
     * @return this.ingredientsList.
     */
    public String[] getIngredientsList() {
        return this.ingredientsList;
    }
}
