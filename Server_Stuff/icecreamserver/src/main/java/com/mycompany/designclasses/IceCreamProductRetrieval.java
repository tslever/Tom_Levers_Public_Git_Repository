
/*
 * This document was created as part of the decomposition of server subsystem
 * IceCreamProductRetrieval.
 */

/*
 * Distinguishes this design class from IceCreamProductRetrieval in
 * package com.mycompany.serverutilities.
 */
package com.mycompany.designclasses;

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