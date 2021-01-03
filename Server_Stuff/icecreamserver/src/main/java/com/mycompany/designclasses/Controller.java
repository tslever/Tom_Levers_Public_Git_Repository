
/*
 * Distinguishes this design class from Controller in package
 * com.mycompany.serverutilities.
 */
package com.mycompany.designclasses;

/**
 * Defines class Controller that defines basic functionality to process a
 * an HTTP message.
 * @version 0.0
 * @author Tom Lever
 */
public class Controller {
    
    /**
     * Defines constructor Controller.
     */
    public Controller() { }
    
    /**
     * Defines method process, which must be overridden to process
     * IceCreamApplicationMessages into Products.
     * @param messageToProcess
     * @throws Exception
     */
    public void process(IceCreamApplicationMessage messageToProcess)
    throws Exception { }
}