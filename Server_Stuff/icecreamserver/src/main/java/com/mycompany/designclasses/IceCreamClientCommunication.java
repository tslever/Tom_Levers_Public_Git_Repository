
/*
 * This document was created as part of the decomposition of server subsystem
 * IceCreamClientCommunication.
 */

/*
 * Distinguishes this design class from IceCreamClientCommunication in
 * package com.mycompany.serverutilities.
 */
package com.mycompany.designclasses;

/**
 * Defines class IceCreamClientCommunication, an instance of which represents
 * an ice cream client communication subsystem and receives search parameters
 * from the ice cream client, works with an instance of IceCreamProductRetrieval
 * to determine the products that match those search parameters, and provides
 * those matching products to the ice cream client.
 * @author Tom Lever
 */
public class IceCreamClientCommunication {
   
    /**
     * Defines constructor IceCreamClientCommunication which sets attributes of
     * this iceCreamClientCommunication with inputs.
     * inputs.
     * @param retrieverToUse
     */
    public IceCreamClientCommunication(
        IceCreamProductRetrieval retrieverToUse) { }
}
