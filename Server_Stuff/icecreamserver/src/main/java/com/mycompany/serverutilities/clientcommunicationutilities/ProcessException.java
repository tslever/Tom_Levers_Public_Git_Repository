/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.serverutilities.clientcommunicationutilities;

/**
 *
 * @author thoma
 */
public class ProcessException extends Exception {
    public ProcessException(String errorMessageToUse) {
        super(errorMessageToUse);
    }
}
