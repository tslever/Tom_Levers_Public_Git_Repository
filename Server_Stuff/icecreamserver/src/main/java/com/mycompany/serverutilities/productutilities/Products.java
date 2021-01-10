
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
    
    /**
     * Defines constructor Products.
     */
    public Products() {
        logger.log(new LogRecord(Level.INFO,
            "Products controller: Started."));
    }
}