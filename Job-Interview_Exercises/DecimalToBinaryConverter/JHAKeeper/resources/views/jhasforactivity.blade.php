@extends('layouts.app')

@section('content')
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-8">

                <a href="/activities">Return to List of Activities with JHA's</a><br/><br/>

                <h1>Job Hazard Analysis for Activity {{$activity_name}}</h1><br/>

                <table width="100%">
                    <tr>
                        <th>Activity Name</th>
                        <th>Job Step</th>
                        <th>Hazard</th>
                        <th>Control</th>
                    </tr>
                    @foreach($jhas as $jha)
                        <tr>
                            <td>{{$jha->activity_name}}</td>
                            <td>{{$jha->job_step}}</td>
                            <td>{{$jha->hazard}}</td>
                            <td>{{$jha->control}}</td>
                        </tr>
                    @endforeach
                </table>

            </div>
        </div>
    </div>
@endsection
