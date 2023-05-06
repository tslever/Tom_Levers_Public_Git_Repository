<?php

namespace App\Http\Controllers;

class JhaFormController extends Controller
{
    /**
     * Create a new controller instance.
     *
     * @return void
     */
    public function __construct()
    {
        $this->middleware('auth');
    }

    /**
     * When client navigates to endpoint '/jha_form',
     * the message interface associated with '/jha_form' calls index.
     * Laravel gets $jhas, a slice of database table jhas,
     * adds HTML returned by Laravel / PHP commands in jhaform.blade.php
     * to the HTML already in jhaform.blade.php,
     * and returns the HTML to the client.
     *
     * @return \Illuminate\Contracts\Support\Renderable
     */
    public function index()
    {
        $jhas = \Illuminate\Support\Facades\DB::table('jhas')
            ->orderBy('activity_name')
            ->orderBy('job_step')
            ->orderBy('hazard')
            ->orderBy('control')
            ->get();
        return view('jhaform', ['jhas' => $jhas, 'activity_name' => ""]);
    }

    public function index2($activity_name) {
        $jhas = \Illuminate\Support\Facades\DB::table('jhas')
            ->orderBy('activity_name')
            ->orderBy('job_step')
            ->orderBy('hazard')
            ->orderBy('control')
            ->get();
        return view('jhaform', ['jhas' => $jhas, 'activity_name' => $activity_name]);
    }

    public function save(\Illuminate\Http\Request $request)
    {
        $jha = new \App\Models\Jha;
        $jha->activity_name = $request->activity_name;
        $jha->job_step = $request->job_step;
        $jha->hazard = $request->hazard;
        $jha->control = $request->control;
        $jha->save();

        return redirect('/jha_form/'.$request->activity_name);
    }
}
