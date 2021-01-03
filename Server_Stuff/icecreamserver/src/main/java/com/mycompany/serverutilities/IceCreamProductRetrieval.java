
// Allows executable to find class IceCreamProductRetrieval.
package com.mycompany.serverutilities;

/**
 * Defines class IceCreamProductRetrieval, an instance of which represents
 * an ice cream product retrieval subsystem and works with an instance of
 * IceCreamClientCommunication to determine the ice cream products that match
 * specific search criteria.
 * @version 0.0
 * @author Tom Lever
 */
public class IceCreamProductRetrieval {
    
    /**
     * Defines constructor IceCreamProductRetrieval.
     */
    public IceCreamProductRetrieval() {
        System.out.println("IceCreamProductRetrieval constructor: Started.");
    }
    
    /**
     * Defines method getTheProductsMatching to get the ice cream products
     * matching specific search criteria.
     * @param searchCriteriaToUse
     * @return new Products()
     */
    public Products getTheProductsMatching(SearchCriteria searchCriteriaToUse) {
        // Functionality to get the products matching the search criteria to
        // use.
        return new Products();
    }
}
