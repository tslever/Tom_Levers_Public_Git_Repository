
// Allows executable to find class IceCreamProductRetrieval.
package com.mycompany.serverutilities.productutilities;

import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * Defines class IceCreamProductRetrieval, an instance of which represents
 * an ice cream product retrieval subsystem and works with an instance of
 * IceCreamClientCommunication to determine the ice cream products that match
 * specific search criteria.
 * @version 0.0
 * @author Tom Lever
 */
public class IceCreamProductRetrieval {
    
    private final static Logger logger =
        Logger.getLogger(IceCreamProductRetrieval.class.getName());
    
    /**
     * Defines constructor IceCreamProductRetrieval.
     */
    public IceCreamProductRetrieval() {
        logger.log(new LogRecord(Level.INFO,
            "IceCreamProductRetrieval constructor: Started."));
    }
    
    /**
     * Defines method getTheProductsMatching to get the ice cream products
     * matching specific search criteria.
     * @param searchCriteriaToUse
     * @return new Products()
     */
    public Products getTheProductsMatching(SearchCriteria searchCriteriaToUse) {
        logger.log(new LogRecord(Level.INFO,
            "IceCreamProductRetrieval.getTheProductsMatching: Started."));
        
        // Functionality to get the products matching the search criteria to
        // use.
      
        logger.log(new LogRecord(Level.INFO,
            "IceCreamProductRetrieval.getTheProductsMatching: Functionality " +
            "to get the products matching the inputted search criteria does " +
            "not exist: Returning a Products with relevant info and an empty " +
            "String array of products."));
        return new Products(
            "Zero products available: Functionality to get products matching " +
            "search criteria does not exist.",
            new String[0]);
    }
}
