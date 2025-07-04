import React, { useState } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";

export default function Repair2SkillApp() {
  const [image, setImage] = useState(null);
  const [repairPlan, setRepairPlan] = useState('');
  const [loading, setLoading] = useState(false);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleGenerateRepairPlan = async () => {
    if (!image) return;
    setLoading(true);
    setRepairPlan('');

    // Simulated GPT-4o response for demonstration
    setTimeout(() => {
      setRepairPlan(`Step 1: Remove the broken left leg.\nStep 2: Insert a new leg.\nStep 3: Reattach the seat panel.\nStep 4: Secure the backrest.`);
      setLoading(false);
    }, 1500);
  };

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Repair2Skill Web Interface</h1>

      <Card>
        <CardContent className="space-y-4 pt-4">
          <Input type="file" accept="image/*" onChange={handleImageUpload} />
          {image && <img src={image} alt="Uploaded chair" className="max-w-md rounded-lg" />}

          <Button onClick={handleGenerateRepairPlan} disabled={!image || loading}>
            {loading ? 'Generating...' : 'Generate Repair Plan'}
          </Button>

          <Textarea readOnly value={repairPlan} rows={6} placeholder="Repair plan will appear here..." />
        </CardContent>
      </Card>
    </div>
  );
}