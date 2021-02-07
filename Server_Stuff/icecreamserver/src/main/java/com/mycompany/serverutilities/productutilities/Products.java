
// Allows executable to find class Products.
package com.mycompany.serverutilities.productutilities;

import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * Defines class Products, an instance of which informs the sending of a message
 * to the ice cream client listing ice cream products (in response to the server
 * receiving search parameters from the ice cream client).
 * @version 0.0
 * @author Tom Lever
 */
public class Products {

    private final static Logger logger =
        Logger.getLogger(Products.class.getName());
    
    String info;
    String[] products;
    
    /**
     * Defines constructor Products.
     */
    public Products(String infoToUse, String[] productsToUse) {
        logger.log(new LogRecord(Level.INFO,
            "Products controller: Started."));
        
        this.info = infoToUse;
        this.products = productsToUse;
    }
    
    public String getInfo() {
        return this.info;
    }
    
    public String[] getProducts() {
        return this.products;
    }
}