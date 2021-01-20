
// Allows executable to find class SearchCriteria.
package com.mycompany.serverutilities.productutilities;

import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * Defines class SearchCriteria, which will be extracted from an ice cream
 * application message and will inform getting the appropriate ice cream
 * products.
 * @version 0.0
 * @author Tom Lever
 */
public class SearchCriteria {

    private final static Logger logger =
        Logger.getLogger(SearchCriteria.class.getName());
    
    private final String[] ingredientsList;
    
    /**
     * Defines constructor SearchCriteria.
     * @param ingredientsListToUse
     */
    public SearchCriteria(String[] ingredientsListToUse) {
        logger.log(new LogRecord(Level.INFO,
            "SearchCriteria constructor: Started."));
        
        this.ingredientsList = ingredientsListToUse;
        logger.log(new LogRecord(Level.INFO,
            "SearchCriteria constructor: Set ingredientsList as " +
            "ingredientsListToUse."));
    }
}
