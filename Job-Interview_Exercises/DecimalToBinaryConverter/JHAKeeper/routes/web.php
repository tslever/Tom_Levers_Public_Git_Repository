<?php
// web.php sets up message interfaces.
// Each get call gets a message interface.
// Each message interface contains an endpoint and a processing.

// Endpoint: '/'.
// Processing: function () { return view('welcome'); }
// When client sends message to endpoint '/', Laravel responds with welcome view.
Illuminate\Support\Facades\Route::get('/', function () {
    return view('welcome');
});

// Auth is adding its own routing information.
Illuminate\Support\Facades\Auth::routes();

// Endpoint: '/home'.
// Processing: function () { $userId = ...; return redirect('/activities'); }
// When client sends message to endpoint '/home', Laravel redirects client to '/activities'.
Illuminate\Support\Facades\Route::get('/home', function () {
    return redirect('/activities');
});

// Endpoint: '/activities'.
// Processing: index of ActivitiesController.
// When client sends a message to endpoint '/activities',
// Laravel calls index of ActivitiesController.
Illuminate\Support\Facades\Route::get('/activities',
    [App\Http\Controllers\ActivitiesController::class, 'index'])->name('activities');

// Endpoint: '/jha_form'.
// Processing: index of JhaFormController.
// When client sends a message to endpoint '/jha_form', Laravel calls index of JhaFormController.
Illuminate\Support\Facades\Route::get('/jha_form',
    [App\Http\Controllers\JhaFormController::class, 'index'])->name('jhaform');

// Endpoint: '/jha_form'.
// Processing: index of JhaFormController.
// When client sends a message to endpoint '/jha_form', Laravel calls index of JhaFormController.
Illuminate\Support\Facades\Route::get('/jha_form/{activity_name}',
    [App\Http\Controllers\JhaFormController::class, 'index2'])->name('jhaform');

// When client submits a JHA, Laravel calls save of JhaFormController.
Illuminate\Support\Facades\Route::post('submit',
    'App\Http\Controllers\JhaFormController@save');

// When client submits a JHA, Laravel calls save of JhaFormController.
Illuminate\Support\Facades\Route::post('/jha_form/submit',
    'App\Http\Controllers\JhaFormController@save');

// When client clicks on a hyperlink in the page associated with '/activities',
// Laravel navigates to '/jha/{activity_name}' and calls index of JhasForActivityController.
Illuminate\Support\Facades\Route::get('/jha/{activity_name}', [
    'as' => 'routeForJHAsForActivity',
    'uses' => '\App\Http\Controllers\JhasForActivityController@index'
]);
