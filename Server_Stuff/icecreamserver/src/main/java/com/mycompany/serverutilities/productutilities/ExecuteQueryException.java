/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.serverutilities.productutilities;

/**
 *
 * @author thoma
 */
public class ExecuteQueryException extends Exception {
    public ExecuteQueryException(String errorMessageToUse) {
        super(errorMessageToUse);
    }
}
